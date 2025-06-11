import pandas
import numpy as np
import xml.etree.cElementTree as ET


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


def define_multibasin_model(
    reference_top: dict[str, pandas.DataFrame],
    reference_xml: str,
    additional_top: dict[str, pandas.DataFrame],
    additional_xml: str,
    idx_from_reference_to_additional: dict[int, int],
    idx_from_additional_to_reference: dict[int, int],
    mode_angles: str = "middle",
) -> dict[str, pandas.DataFrame]:
    """
    Updates the dihedrals and angles from the reference topology to the middle value between the two topologies.
    """

    multibasin_top = reference_top.copy()
    additional_top_in_reference_idx = {}

    reference_xml = ET.parse(reference_xml)
    reference_root = reference_xml.getroot()

    additional_xml = ET.parse(additional_xml)
    additional_root = additional_xml.getroot()

    # For the angles
    angles_converted_to_reference = {
        col: [] for col in additional_top["angles"].columns
    }

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

    merged_angles = pandas.merge(
        reference_top["angles"],
        additional_top,
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
        merged_angles["_merge"] == "left_only", "th0(deg)_2"
    ]

    if mode_angles == "middle":
        merged_angles["th0(deg)"] = np.nanmean(
            merged_angles[["th0(deg)_1", "th0(deg)_2"]],
            axis=1,
        )

        multibasin_top["angles"] = merged_angles[
            ["ai", "aj", "ak", "func", "th0(deg)", "Ka"]
        ].reset_index(drop=True)

    elif mode_angles == "flat_bottom":
        angles_xml = ET.SubElement(reference_root, "angles")
        flat_bottom_xml = ET.SubElement(
            angles_xml, "angles_type", attrib={"name": "flat_bottom"}
        )
        ET.SubElement(
            flat_bottom_xml,
            "expression",
            attrib={
                "expr": " Ka * ((theta1 - theta)^2 * step(theta1 - theta) + (theta - theta2)^2 * step(theta-theta2))"
            },
        )

    return pandas.DataFrame(angles_converted_to_reference)
