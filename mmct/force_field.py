import copy
import xml.etree.cElementTree as ET

import numpy as np
import pandas


def read_top(topfile: str) -> dict[str, pandas.DataFrame]:
    """
    Reads a GROMACS topology file and extracts force field parameters.

    Args:
        topfile (str): Path to the GROMACS topology file.

    Returns:
        dict: A dictionary containing force field parameters.
    """
    with open(topfile, "r") as f:
        lines = f.readlines()

    ff_params = {}
    headers = {}
    current_section = None

    treat_comment_as_header = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        elif line.startswith(";"):
            if treat_comment_as_header:
                # Treat this comment as a section header
                headers[current_section] = line[1:].split()
                treat_comment_as_header = False

        elif line.startswith("["):
            current_section = line[1:-1].strip()  # Get section name
            ff_params[current_section] = []
            treat_comment_as_header = True

        elif current_section:
            ff_params[current_section].append(line.split())

    for section in ff_params.keys():
        ff_params[section] = pandas.DataFrame(
            data=ff_params[section],
            columns=headers[section][: len(ff_params[section][0])],
        )

    # Convert specific columns to appropriate types
    if "bonds" in ff_params:
        ff_params["bonds"][["ai", "aj", "func"]] = ff_params["bonds"][
            ["ai", "aj", "func"]
        ].astype(int)
        ff_params["bonds"][["r0(nm)", "Kb"]] = ff_params["bonds"][
            ["r0(nm)", "Kb"]
        ].astype(float)
    if "angles" in ff_params:
        ff_params["angles"][["ai", "aj", "ak", "func"]] = ff_params[
            "angles"
        ][["ai", "aj", "ak", "func"]].astype(int)
        ff_params["angles"][["th0(deg)", "Ka"]] = ff_params["angles"][
            ["th0(deg)", "Ka"]
        ].astype(float)

    return ff_params


def save_top(
    top: dict[str, pandas.DataFrame], topfile: str = "top.top"
) -> None:
    """
    Saves force field parameters to a GROMACS topology file.

    Args:
        top (dict): A dictionary containing force field parameters.
        topfile (str): Path to the output GROMACS topology file.
    """

    def format_entry(val):
        if isinstance(val, int):
            return str(val)
        elif isinstance(val, float):
            return f"{val:.9e}"
        else:
            return str(val)

    with open(topfile, "w") as f:
        for section, df in top.items():
            f.write(f"[{section}]\n")
            f.write("; " + "\t".join(df.columns) + "\n")
            for row in df.itertuples(index=False):
                f.write(
                    "\t".join(format_entry(x) for x in row) + "\n"
                )
            f.write("\n")


def save_xml(
    xml_tree: ET.ElementTree, xmlfile: str = "top.xml"
) -> None:
    """
    Saves an XML tree to a file.

    Args:
        xml_tree (ET.ElementTree): The XML tree to save.
        xmlfile (str): Path to the output XML file.
    """
    ET.indent(xml_tree, space=" ")
    xml_tree.write(xmlfile)


def _process_angles(
    reference_top: dict[str, pandas.DataFrame],
    additional_top: dict[str, pandas.DataFrame],
    multibasin_top: dict[str, pandas.DataFrame],
    idx_from_additional_to_reference: dict[int, int],
    multibasin_xml: ET.ElementTree,
    mode_angles: str,
):
    """
    Processes angles from the reference and additional topologies,
    converting them to the reference indices and merging them.
    The angles are either averaged or converted to a flat-bottom potential.
    """

    angles_converted_to_reference = {
        col: [] for col in additional_top["angles"].columns
    }

    reference_root = multibasin_xml.getroot()

    for _, row in additional_top["angles"].iterrows():

        i = idx_from_additional_to_reference.get(row["ai"])
        j = idx_from_additional_to_reference.get(row["aj"])
        k = idx_from_additional_to_reference.get(row["ak"])

        if i is not None and j is not None and k is not None:
            angles_converted_to_reference["ai"].append(i)
            angles_converted_to_reference["aj"].append(j)
            angles_converted_to_reference["ak"].append(k)
            angles_converted_to_reference["func"].append(row["func"])
            angles_converted_to_reference["th0(deg)"].append(
                row["th0(deg)"]
            )
            angles_converted_to_reference["Ka"].append(row["Ka"])

    df_angles_additional = pandas.DataFrame(
        data=angles_converted_to_reference
    )
    df_angles_additional[["ai", "aj", "ak", "func"]] = (
        df_angles_additional[["ai", "aj", "ak", "func"]].astype(int)
    )
    df_angles_additional[["th0(deg)", "Ka"]] = df_angles_additional[
        ["th0(deg)", "Ka"]
    ].astype(float)

    merged_angles = pandas.merge(
        reference_top["angles"],
        df_angles_additional,
        left_on=["ai", "aj", "ak", "func", "Ka"],
        right_on=["ai", "aj", "ak", "func", "Ka"],
        how="outer",
        suffixes=("_1", "_2"),
        indicator=True,
    )

    merged_angles.loc[
        merged_angles["_merge"] == "left_only", "th0(deg)_2"
    ] = merged_angles.loc[
        merged_angles["_merge"] == "left_only", "th0(deg)_1"
    ]
    merged_angles.loc[
        merged_angles["_merge"] == "right_only", "th0(deg)_1"
    ] = merged_angles.loc[
        merged_angles["_merge"] == "right_only", "th0(deg)_2"
    ]

    if mode_angles == "middle":
        merged_angles["th0(deg)"] = np.nanmean(
            merged_angles[["th0(deg)_1", "th0(deg)_2"]],
            axis=1,
        )

        multibasin_top["angles"] = merged_angles[
            ["ai", "aj", "ak", "func", "th0(deg)", "Ka"]
        ].reset_index(drop=True)

        return multibasin_top, multibasin_xml

    elif mode_angles == "flat_bottom":
        angles_xml = ET.SubElement(reference_root, "angles")
        flat_bottom_xml = ET.SubElement(
            angles_xml, "angles_type", attrib={"name": "flat_bottom"}
        )
        ET.SubElement(
            flat_bottom_xml,
            "expression",
            attrib={
                "expr": "Ka * ((theta1 - theta)^2 * step(theta1 - theta) + (theta - theta2)^2 * step(theta-theta2))"
            },
        )
        ET.SubElement(flat_bottom_xml, "parameter").text = "Ka"
        ET.SubElement(flat_bottom_xml, "parameter").text = "theta1"
        ET.SubElement(flat_bottom_xml, "parameter").text = "theta2"

        for _, row in merged_angles.iterrows():

            theta1 = min(row["th0(deg)_1"], row["th0(deg)_2"])
            theta2 = max(row["th0(deg)_1"], row["th0(deg)_2"])

            ET.SubElement(
                flat_bottom_xml,
                "interaction",
                attrib={
                    "i": str(row["ai"]),
                    "j": str(row["aj"]),
                    "k": str(row["ak"]),
                    "Ka": f"{row["Ka"]:.5e}",
                    "theta1": f"{theta1:.5e}",
                    "theta2": f"{theta2:.5e}",
                },
            )

        del multibasin_top["angles"]
        return multibasin_top, multibasin_xml
    else:
        raise ValueError(
            f"Unknown mode for angles: {mode_angles}. "
            "Supported modes are 'middle' and 'flat_bottom'."
        )


def _process_bonds(
    reference_top: dict[str, pandas.DataFrame],
    additional_top: dict[str, pandas.DataFrame],
    multibasin_top: dict[str, pandas.DataFrame],
    idx_from_additional_to_reference: dict[int, int],
    multibasin_xml: ET.ElementTree,
):
    """
    Processes bonds from the reference and additional topologies,
    converting them to the reference indices and merging them.
    The final bonds are averages of the equilibrium length in both
    structures.
    """

    bonds_converted_to_reference = {
        col: [] for col in additional_top["bonds"].columns
    }

    reference_root = multibasin_xml.getroot()

    for _, row in additional_top["bonds"].iterrows():

        i = idx_from_additional_to_reference.get(row["ai"])
        j = idx_from_additional_to_reference.get(row["aj"])

        if i is not None and j is not None:
            bonds_converted_to_reference)
