import copy
import xml.etree.ElementTree as ET

import numpy as np
import pandas

from . import pdb_tools


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
            ff_params[current_section].append(
                line.split(sep=";")[0].split()
            )

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
        try:
            ff_params["bonds"][["r0(nm)", "Kb"]] = ff_params["bonds"][
                ["r0(nm)", "Kb"]
            ].astype(float)
        except KeyError:
            print(
                "Warning: 'r0(nm)' or 'Kb' column not found in bonds section.",
                flush=True,
            )
            print(
                "Make sure 'bondtypes' are defined in the topology file.",
                flush=True,
            )
    if "angles" in ff_params:
        ff_params["angles"][["ai", "aj", "ak", "func"]] = ff_params[
            "angles"
        ][["ai", "aj", "ak", "func"]].astype(int)
        try:
            ff_params["angles"][["th0(deg)", "Ka"]] = ff_params[
                "angles"
            ][["th0(deg)", "Ka"]].astype(float)
        except KeyError:
            print(
                "Warning: 'th0(deg)' or 'Ka' column not found in angles section.",
                flush=True,
            )
            print(
                "Make sure 'angletypes' are defined in the topology file.",
                flush=True,
            )

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
            f.write(f"[ {section} ]\n")
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


def _remove_xml_section(tree: ET.ElementTree, path: str):
    """
    Remove *every* element matching `path` (and all its children)
    from the XML tree—without using O(N^2) remove() calls.
    """
    root = tree.getroot()
    if root is None:
        raise ValueError("The XML tree is empty or malformed.")

    # 1. Find all elements to remove
    to_remove_list = root.findall(path)
    to_remove_set = set(to_remove_list)

    # 2. Build a parent map (child → parent) for the whole tree
    parent_map = {c: p for p in tree.iter() for c in p}

    # 3. Group removals by their parent
    parents = {}
    for elem in to_remove_list:
        parent = parent_map.get(elem)
        if parent is not None:
            parents.setdefault(parent, []).append(elem)

    # 4. For each affected parent, rebuild its children list in one go
    for parent in parents:
        # Keep only those children not slated for removal
        kept_children = [
            child for child in parent if child not in to_remove_set
        ]
        parent.clear()  # drop all old children in one C-call
        parent.extend(kept_children)  # add back only the survivors

    return tree


def _process_angles(
    reference_top: dict[str, pandas.DataFrame],
    additional_top: dict[str, pandas.DataFrame],
    multibasin_top: dict[str, pandas.DataFrame],
    idx_from_additional_to_reference: dict[int, int],
    multibasin_xml: ET.ElementTree,
    mode: str = "middle",
) -> tuple[dict[str, pandas.DataFrame], ET.ElementTree]:
    """
    Processes angles from the reference and additional topologies,
    converting them to the reference indices and merging them.
    The angles are either averaged or converted to a flat-bottom potential.
    """

    angles_converted_to_reference = {
        col: [] for col in additional_top["angles"].columns
    }

    multibasin_root = multibasin_xml.getroot()
    if multibasin_root is None:
        raise ValueError("The XML tree is empty or malformed.")

    for _, row in additional_top["angles"].iterrows():

        i = idx_from_additional_to_reference.get(row["ai"])
        j = idx_from_additional_to_reference.get(row["aj"])
        k = idx_from_additional_to_reference.get(row["ak"])

        if i is not None and j is not None and k is not None:
            angles_converted_to_reference["ai"].append(i)
            angles_converted_to_reference["aj"].append(j)
            angles_converted_to_reference["ak"].append(k)
            angles_converted_to_reference["func"].append(row["func"])

            if "th0(deg)" in row and "Ka" in row:
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
    if (
        "th0(deg)" in df_angles_additional.columns
        and "Ka" in df_angles_additional.columns
    ):
        df_angles_additional[["th0(deg)", "Ka"]] = (
            df_angles_additional[["th0(deg)", "Ka"]].astype(float)
        )

    if "angles" in reference_top:
        df_angles_reference = reference_top["angles"].copy()
    else:
        # angles information present in XML
        angles_reference = {
            "ai": [],
            "aj": [],
            "ak": [],
            "func": [],
            "theta1": [],
            "theta2": [],
            "Ka": [],
        }
        for angle_type in multibasin_root.findall(
            ".//angles/angles_type"
        ):
            for interaction in angle_type.findall("interaction"):
                i = int(interaction.attrib["i"])
                j = int(interaction.attrib["j"])
                k = int(interaction.attrib["k"])

                angles_reference["ai"].append(i)
                angles_reference["aj"].append(j)
                angles_reference["ak"].append(k)
                angles_reference["func"].append(1)
                angles_reference["theta1"].append(
                    np.degrees(float(interaction.attrib["theta1"]))
                )
                angles_reference["theta2"].append(
                    np.degrees(float(interaction.attrib["theta2"]))
                )
                angles_reference["Ka"].append(
                    float(interaction.attrib["Ka"])
                )
        df_angles_reference = pandas.DataFrame(data=angles_reference)
        df_angles_reference[["ai", "aj", "ak", "func"]] = (
            df_angles_reference[["ai", "aj", "ak", "func"]].astype(
                int
            )
        )
        df_angles_reference[["theta1", "theta2", "Ka"]] = (
            df_angles_reference[["theta1", "theta2", "Ka"]].astype(
                float
            )
        )
        df_angles_reference["th0(deg)"] = np.nanmean(
            df_angles_reference[["theta1", "theta2"]],
            axis=1,
        )
        df_angles_reference["editable"] = True
        df_angles_reference.loc[
            df_angles_reference["theta1"]
            != df_angles_reference["theta2"],
            "editable",
        ] = False

        df_angles_reference.drop(
            ["theta1", "theta2"], axis=1, inplace=True
        )

    merged_angles = pandas.merge(
        df_angles_reference,
        df_angles_additional,
        left_on=["ai", "aj", "ak", "func"],
        right_on=["ai", "aj", "ak", "func"],
        how="outer",
        suffixes=("_1", "_2"),
        indicator="source",
    )

    if "Ka" in df_angles_additional.columns:
        merged_angles["Ka"] = np.nanmax(
            merged_angles[["Ka_1", "Ka_2"]],
            axis=1,
        )

    if "editable" in df_angles_reference.columns:
        # check if any of the angles in both structures are not editable
        if len(
            merged_angles.loc[
                (merged_angles["source"] == "both")
                & (merged_angles["editable"] == False)
            ]
        ):
            raise ValueError(
                "Angles already edited before is trying to be edited again. "
                "This is not supported. "
            )

    if "th0(deg)" in df_angles_additional.columns:
        merged_angles.loc[
            merged_angles["source"] == "left_only", "th0(deg)_2"
        ] = merged_angles.loc[
            merged_angles["source"] == "left_only", "th0(deg)_1"
        ]
        merged_angles.loc[
            merged_angles["source"] == "right_only", "th0(deg)_1"
        ] = merged_angles.loc[
            merged_angles["source"] == "right_only", "th0(deg)_2"
        ]

    if mode == "middle":
        merged_angles["th0(deg)"] = np.nanmean(
            merged_angles[["th0(deg)_1", "th0(deg)_2"]],
            axis=1,
        )

        multibasin_top["angles"] = merged_angles[
            ["ai", "aj", "ak", "func", "th0(deg)", "Ka"]
        ].reset_index(drop=True)

        return multibasin_top, multibasin_xml

    elif mode == "flat_bottom":

        # remove angles from multibasin_xml if they exist
        multibasin_xml = _remove_xml_section(
            multibasin_xml, ".//angles"
        )

        angles_xml = ET.SubElement(multibasin_root, "angles")
        flat_bottom_xml = ET.SubElement(
            angles_xml,
            "angles_type",
            attrib={"name": "angle_flat_bottom"},
        )
        ET.SubElement(
            flat_bottom_xml,
            "expression",
            attrib={
                "expr": "Ka / 2 * ((theta1 - theta)^2 * step(theta1 - theta) + (theta - theta2)^2 * step(theta-theta2))"
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
                    "Ka": f"{row['Ka']:.5e}",
                    "theta1": f"{np.radians(theta1):.5e}",
                    "theta2": f"{np.radians(theta2):.5e}",
                },
            )

        if "angles" in multibasin_top:
            del multibasin_top["angles"]

        return multibasin_top, multibasin_xml

    elif mode == "amber":
        multibasin_top["angletypes"] = pandas.merge(
            left=reference_top["angletypes"],
            right=additional_top["angletypes"],
            how="inner",
            indicator=False,
        ).reset_index(drop=True)

        multibasin_top["angles"] = merged_angles[
            ["ai", "aj", "ak", "func"]
        ].reset_index(drop=True)

        return multibasin_top, multibasin_xml

    else:
        raise ValueError(
            f"Unknown mode for angles: {mode}. "
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

    multibasin_root = multibasin_xml.getroot()

    for _, row in additional_top["bonds"].iterrows():

        i = idx_from_additional_to_reference.get(row["ai"])
        j = idx_from_additional_to_reference.get(row["aj"])

        if i is not None and j is not None:
            bonds_converted_to_reference["ai"].append(i)
            bonds_converted_to_reference["aj"].append(j)
            bonds_converted_to_reference["func"].append(row["func"])
            if "r0(nm)" in row and "Kb" in row:
                bonds_converted_to_reference["r0(nm)"].append(
                    row["r0(nm)"]
                )
                bonds_converted_to_reference["Kb"].append(row["Kb"])

    df_bonds_additional = pandas.DataFrame(
        data=bonds_converted_to_reference
    )
    df_bonds_additional[["ai", "aj", "func"]] = df_bonds_additional[
        ["ai", "aj", "func"]
    ].astype(int)
    if ("r0(nm)" in df_bonds_additional.columns) and (
        "Kb" in df_bonds_additional.columns
    ):
        df_bonds_additional[["r0(nm)", "Kb"]] = df_bonds_additional[
            ["r0(nm)", "Kb"]
        ].astype(float)

    merged_bonds = pandas.merge(
        reference_top["bonds"],
        df_bonds_additional,
        left_on=["ai", "aj", "func"],
        right_on=["ai", "aj", "func"],
        how="outer",
        suffixes=("_1", "_2"),
        indicator="source",
    )

    if "Kb" in df_bonds_additional.columns:
        merged_bonds["Kb"] = np.nanmax(
            merged_bonds[["Kb_1", "Kb_2"]],
            axis=1,
        )

    if "r0(nm)" in df_bonds_additional.columns:
        merged_bonds.loc[
            merged_bonds["source"] == "left_only", "r0(nm)_2"
        ] = merged_bonds.loc[
            merged_bonds["source"] == "left_only", "r0(nm)_1"
        ]
        merged_bonds.loc[
            merged_bonds["source"] == "right_only", "r0(nm)_1"
        ] = merged_bonds.loc[
            merged_bonds["source"] == "right_only", "r0(nm)_2"
        ]

        merged_bonds["r0(nm)"] = np.nanmean(
            merged_bonds[["r0(nm)_1", "r0(nm)_2"]],
            axis=1,
        )

    multibasin_top["bonds"] = merged_bonds[
        list(df_bonds_additional.columns)
    ].reset_index(drop=True)

    if "bondtypes" in reference_top or "bondtypes" in additional_top:
        multibasin_top["bondtypes"] = pandas.merge(
            left=reference_top.get("bondtypes", pandas.DataFrame()),
            right=additional_top.get("bondtypes", pandas.DataFrame()),
            how="inner",
            indicator=False,
        ).reset_index(drop=True)

    return multibasin_top, multibasin_xml


def _normalize_angle(angles: np.ndarray) -> np.ndarray:
    """
    Wrap angle(s) into the interval [-π, π).
    Works with scalars or array-like inputs.
    """
    angles = np.asarray(angles)
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _angular_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the circular midpoint of a and b (radians, assumed in [-π,π]).
    Returns angle(s) in [-π, π). Accepts scalar or array inputs.
    """
    a = _normalize_angle(a)
    b = _normalize_angle(b)

    # vector sum
    x = np.cos(a) + np.cos(b)
    y = np.sin(a) + np.sin(b)
    mid = np.arctan2(y, x)

    return _normalize_angle(mid)


def _process_dihedrals(
    reference_xml: ET.ElementTree,
    additional_xml: ET.ElementTree,
    multibasin_xml: ET.ElementTree,
    idx_from_additional_to_reference: dict[int, int],
) -> ET.ElementTree:
    """
    Processes dihedrals from the reference and additional XML files,
    converting them to the reference indices and merging them.
    The final dihedrals are averages of the parameters in both structures.
    """

    reference_root = reference_xml.getroot()
    if reference_root is None:
        raise ValueError(
            "The reference XML tree is empty or malformed."
        )
    additional_root = additional_xml.getroot()
    if additional_root is None:
        raise ValueError(
            "The additional XML tree is empty or malformed."
        )
    multibasin_root = multibasin_xml.getroot()
    if multibasin_root is None:
        raise ValueError(
            "The multibasin XML tree is empty or malformed."
        )

    dihedrals_converted_to_reference = {}
    df_additional = {}

    for dihedral_type in additional_root.findall(
        ".//dihedrals/dihedrals_type"
    ):
        dihedral_type_name = dihedral_type.get("name")

        dihedrals_converted_to_reference[dihedral_type_name] = {
            col: []
            for col in dihedral_type.findall("interaction")[
                0
            ].attrib.keys()
        }

        no_indices_columns = []
        for col in dihedrals_converted_to_reference[
            dihedral_type_name
        ]:
            if col not in ["i", "j", "k", "l"]:
                no_indices_columns.append(col)

        for interaction in dihedral_type.findall("interaction"):
            # Convert dihedral indices from additional to reference
            i = idx_from_additional_to_reference.get(
                int(interaction.attrib["i"])
            )
            j = idx_from_additional_to_reference.get(
                int(interaction.attrib["j"])
            )
            k = idx_from_additional_to_reference.get(
                int(interaction.attrib["k"])
            )
            l = idx_from_additional_to_reference.get(
                int(interaction.attrib["l"])
            )

            if (
                i is not None
                and j is not None
                and k is not None
                and l is not None
            ):
                dihedrals_converted_to_reference[dihedral_type_name][
                    "i"
                ].append(i)
                dihedrals_converted_to_reference[dihedral_type_name][
                    "j"
                ].append(j)
                dihedrals_converted_to_reference[dihedral_type_name][
                    "k"
                ].append(k)
                dihedrals_converted_to_reference[dihedral_type_name][
                    "l"
                ].append(l)

                for key in no_indices_columns:
                    dihedrals_converted_to_reference[
                        dihedral_type_name
                    ][key].append(interaction.attrib[key])

        df_additional[dihedral_type_name] = pandas.DataFrame(
            data=dihedrals_converted_to_reference[dihedral_type_name]
        )
        df_additional[dihedral_type_name][["theta0", "weight"]] = (
            df_additional[dihedral_type_name][
                ["theta0", "weight"]
            ].astype(float)
        )
        df_additional[dihedral_type_name][["i", "j", "k", "l"]] = (
            df_additional[dihedral_type_name][
                ["i", "j", "k", "l"]
            ].astype(int)
        )
        if (
            "multiplicity"
            in df_additional[dihedral_type_name].columns
        ):
            df_additional[dihedral_type_name][["multiplicity"]] = (
                df_additional[dihedral_type_name][
                    ["multiplicity"]
                ].astype(int)
            )

    dihedrals_in_reference = {}
    df_reference = {}

    for dihedral_type in reference_root.findall(
        ".//dihedrals/dihedrals_type"
    ):
        dihedral_type_name = dihedral_type.get("name")

        dihedrals_in_reference[dihedral_type_name] = {
            col: []
            for col in dihedral_type.findall("interaction")[
                0
            ].attrib.keys()
        }

        no_indices_columns = []
        for col in dihedrals_in_reference[dihedral_type_name]:
            if col not in ["i", "j", "k", "l"]:
                no_indices_columns.append(col)

        for interaction in dihedral_type.findall("interaction"):
            i = int(interaction.attrib["i"])
            j = int(interaction.attrib["j"])
            k = int(interaction.attrib["k"])
            l = int(interaction.attrib["l"])

            dihedrals_in_reference[dihedral_type_name]["i"].append(i)
            dihedrals_in_reference[dihedral_type_name]["j"].append(j)
            dihedrals_in_reference[dihedral_type_name]["k"].append(k)
            dihedrals_in_reference[dihedral_type_name]["l"].append(l)

            for key in no_indices_columns:
                dihedrals_in_reference[dihedral_type_name][
                    key
                ].append(interaction.attrib[key])

        df_reference[dihedral_type_name] = pandas.DataFrame(
            data=dihedrals_in_reference[dihedral_type_name]
        )
        df_reference[dihedral_type_name][["theta0", "weight"]] = (
            df_reference[dihedral_type_name][
                ["theta0", "weight"]
            ].astype(float)
        )
        df_reference[dihedral_type_name][["i", "j", "k", "l"]] = (
            df_reference[dihedral_type_name][
                ["i", "j", "k", "l"]
            ].astype(int)
        )
        if "multiplicity" in df_reference[dihedral_type_name].columns:
            df_reference[dihedral_type_name][["multiplicity"]] = (
                df_reference[dihedral_type_name][
                    ["multiplicity"]
                ].astype(int)
            )

    for dihedral_type in multibasin_root.findall(
        f".//dihedrals/dihedrals_type"
    ):
        dihedral_type_name = dihedral_type.get("name")

        if (
            "multiplicity"
            in df_reference.get(
                dihedral_type_name, pandas.DataFrame()
            ).columns
            and "multiplicity"
            in df_additional.get(
                dihedral_type_name, pandas.DataFrame()
            ).columns
        ):
            merge_on = ["i", "j", "k", "l", "multiplicity"]
        else:
            merge_on = ["i", "j", "k", "l"]

        # Merge the two DataFrames on i, j, k, l
        merged_dihedrals = pandas.merge(
            df_reference[dihedral_type_name],
            df_additional[dihedral_type_name],
            on=merge_on,
            how="outer",
            suffixes=("_1", "_2"),
            indicator="source",
        )

        merged_dihedrals["weight"] = np.nanmin(
            merged_dihedrals[["weight_1", "weight_2"]],
            axis=1,
        )
        print(
            "\n\tWarning: for AA models, only `dihedral_cosine` are "
            "being re-weighted. The other dihedrals will use the `min` "
            "weight. Will need to adjust that for planar dihedrals later",
            flush=True,
        )

        merged_dihedrals.loc[
            merged_dihedrals["source"] == "left_only", "theta0_2"
        ] = merged_dihedrals.loc[
            merged_dihedrals["source"] == "left_only", "theta0_1"
        ]
        merged_dihedrals.loc[
            merged_dihedrals["source"] == "right_only", "theta0_1"
        ] = merged_dihedrals.loc[
            merged_dihedrals["source"] == "right_only", "theta0_2"
        ]

        merged_dihedrals["theta0"] = _angular_midpoint(
            merged_dihedrals["theta0_1"].to_numpy(),
            merged_dihedrals["theta0_2"].to_numpy(),
        )

        # 1) Build a new list of just the non-interaction children
        edited_dihedrals = [
            element
            for element in dihedral_type
            if element.tag != "interaction"
        ]

        # 2) Append your merged interactions
        for row in merged_dihedrals.itertuples(index=False):
            element = ET.Element(
                "interaction",
                attrib={
                    "i": str(row.i),
                    "j": str(row.j),
                    "k": str(row.k),
                    "l": str(row.l),
                    "theta0": f"{row.theta0:.5e}",
                    "weight": (
                        f"{row.weight:.5e}"
                        if row.weight != 1
                        else "1"
                    ),
                },
            )
            if "multiplicity" in merge_on:
                element.set("multiplicity", str(row.multiplicity))

            edited_dihedrals.append(element)

        # 3) In one go, reassign the children of dihedral_type
        dihedral_type[:] = edited_dihedrals

    return multibasin_xml


def _count_dihedrals(
    reference_pdb: pandas.DataFrame, xml: ET.ElementTree
) -> tuple[dict[str, int], pandas.DataFrame]:
    """
    Counts the number of protein and nucleic dihedrals in the force field.

    Args:
        reference_pdb (pandas.DataFrame): DataFrame containing the reference PDB data.
        xml (ET.ElementTree): The XML tree containing the force field parameters.
    Returns:
        tuple: A tuple containing the number of protein dihedrals and
               the number of nucleic dihedrals.
    """

    xml_root = xml.getroot()
    if xml_root is None:
        raise ValueError("The XML tree is empty or malformed.")

    dihedral_cosine_data = {
        "i": [],
        "j": [],
        "k": [],
        "l": [],
        "theta0": [],
        "weight": [],
    }
    for dihedral_coside in xml_root.findall(
        ".//dihedrals/dihedrals_type[@name='dihedral_cosine']/interaction"
    ):
        # only count the dihedrals with multiplicity 1
        if dihedral_coside.attrib["multiplicity"] == "1":
            dihedral_cosine_data["i"].append(
                int(dihedral_coside.attrib["i"])
            )
            dihedral_cosine_data["j"].append(
                int(dihedral_coside.attrib["j"])
            )
            dihedral_cosine_data["k"].append(
                int(dihedral_coside.attrib["k"])
            )
            dihedral_cosine_data["l"].append(
                int(dihedral_coside.attrib["l"])
            )
            dihedral_cosine_data["theta0"].append(
                float(dihedral_coside.attrib["theta0"])
            )
            dihedral_cosine_data["weight"].append(
                float(dihedral_coside.attrib["weight"])
            )

    df_dihedral_cosine = pandas.DataFrame(data=dihedral_cosine_data)
    df_dihedral_cosine["group"] = df_dihedral_cosine.groupby(
        ["j", "k"]
    )["j"].transform("size")

    df_dihedral_cosine["group_weight"] = (
        1 / df_dihedral_cosine["group"]
    )

    df_dihedral_cosine["residue_i"] = reference_pdb.loc[
        df_dihedral_cosine["i"], "residue"
    ].values

    df_dihedral_cosine["molecule_type"] = "protein"

    # fmt: off
    nucleic_residues = [
        "A", "C", "G", "U", 
        "AM", "CM", "GM", "UM", 
        "A0P", "C0P", "G0P", "U0P",
        "A0PM", "C0PM", "G0PM", "U0PM",
    ]
    # fmt: on

    df_dihedral_cosine.loc[
        df_dihedral_cosine["residue_i"].isin(nucleic_residues),
        "molecule_type",
    ] = "nucleic"

    df_dihedral_cosine["atom_i"] = reference_pdb.loc[
        df_dihedral_cosine["i"], "atom"
    ].values
    df_dihedral_cosine["atom_j"] = reference_pdb.loc[
        df_dihedral_cosine["j"], "atom"
    ].values
    df_dihedral_cosine["atom_k"] = reference_pdb.loc[
        df_dihedral_cosine["k"], "atom"
    ].values
    df_dihedral_cosine["atom_l"] = reference_pdb.loc[
        df_dihedral_cosine["l"], "atom"
    ].values

    protein_backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    nucleic_backbone_atoms = {
        # fmt: off
        "P", "O1P", "O2P", "O5*", "C5*", "C4*", "O4*", 
        "C3*", "O3*", "C2*", "O2*", "C1*", "O3P",
        # fmt: on
    }

    df_dihedral_cosine["dihedral_type"] = "sc"
    df_dihedral_cosine.loc[
        df_dihedral_cosine[["atom_i", "atom_j", "atom_k", "atom_l"]]
        .isin(protein_backbone_atoms)
        .all(axis=1),
        "dihedral_type",
    ] = "bb"
    df_dihedral_cosine.loc[
        df_dihedral_cosine[["atom_i", "atom_j", "atom_k", "atom_l"]]
        .isin(nucleic_backbone_atoms)
        .all(axis=1),
        "dihedral_type",
    ] = "bb"

    dihedral_numbers = {
        "protein_bb": df_dihedral_cosine.loc[
            (df_dihedral_cosine["molecule_type"] == "protein")
            & (df_dihedral_cosine["dihedral_type"] == "bb"),
            "group_weight",
        ].sum(),
        "protein_sc": df_dihedral_cosine.loc[
            (df_dihedral_cosine["molecule_type"] == "protein")
            & (df_dihedral_cosine["dihedral_type"] == "sc"),
            "group_weight",
        ].sum(),
        "nucleic_bb": df_dihedral_cosine.loc[
            (df_dihedral_cosine["molecule_type"] == "nucleic")
            & (df_dihedral_cosine["dihedral_type"] == "bb"),
            "group_weight",
        ].sum(),
        "nucleic_sc": df_dihedral_cosine.loc[
            (df_dihedral_cosine["molecule_type"] == "nucleic")
            & (df_dihedral_cosine["dihedral_type"] == "sc"),
            "group_weight",
        ].sum(),
        "total": len(df_dihedral_cosine),
    }

    return dihedral_numbers, df_dihedral_cosine


def _update_dihedral_weights(
    multibasin_xml: ET.ElementTree,
    df_dihedral_cosine: pandas.DataFrame,
) -> ET.ElementTree:
    """
    Updates the dihedral parameters in the multibasin XML tree
    based on the provided dihedral DataFrame and epsilon values.

    Args:
        multibasin_xml (ET.ElementTree): The XML tree to update.
        df_dihedral_cosine (pandas.DataFrame): DataFrame containing dihedral information.
        epsilon_bb (float): Epsilon value for backbone dihedrals.
        epsilon_sc_protein (float): Epsilon value for protein sidechain dihedrals.
        epsilon_sc_nucleic (float): Epsilon value for nucleic sidechain dihedrals.

    Returns:
        ET.ElementTree: The updated XML tree.
    """

    multibasin_root = multibasin_xml.getroot()
    if multibasin_root is None:
        raise ValueError(
            "The multibasin XML tree is empty or malformed."
        )

    df_dihedral_cosine = df_dihedral_cosine.set_index(
        ["i", "j", "k", "l"]
    )
    df_dihedral_cosine["updated_weight"] = (
        df_dihedral_cosine["group_weight"]
        * df_dihedral_cosine["epsilon"]
    )

    for dihedral_type in multibasin_root.findall(
        f".//dihedrals/dihedrals_type/[@name='dihedral_cosine']"
    ):
        modified_dihedrals = []

        for element in dihedral_type:
            if element.tag != "interaction":
                modified_dihedrals.append(element)
            else:
                i = int(element.attrib["i"])
                j = int(element.attrib["j"])
                k = int(element.attrib["k"])
                l = int(element.attrib["l"])

                row = df_dihedral_cosine.loc[(i, j, k, l)]

                if float(element.attrib["multiplicity"]) == 3:
                    updated_weight = 0.5 * row["updated_weight"]
                else:
                    updated_weight = row["updated_weight"]

                element.attrib["weight"] = f"{updated_weight:.5e}"
                modified_dihedrals.append(element)

        dihedral_type[:] = modified_dihedrals

    return multibasin_xml


def _process_contacts(
    reference_xml: ET.ElementTree,
    reference_pdb: pandas.DataFrame,
    additional_xml: ET.ElementTree,
    additional_pdb: pandas.DataFrame,
    multibasin_xml: ET.ElementTree,
    idx_from_additional_to_reference: dict[int, int],
    mode="CG",
    scale_distances=1.0,
) -> tuple[ET.ElementTree, pandas.DataFrame]:
    """
    Processes contacts from the reference and additional XML files,
    converting them to the reference indices and merging them.
    The final contacts are averages of the parameters in both structures.
    """

    if mode == "CG":
        epsilon = 1.0
        POW_ATTRACTION = 10
        COEFF_ATTRACTION = 6
        POW_REPULSION = 12
        COEFF_REPULSION = 5
    elif mode == "AA":
        epsilon = 1.0  # will be adjusted later
        POW_ATTRACTION = 6
        COEFF_ATTRACTION = 2
        POW_REPULSION = 12
        COEFF_REPULSION = 1

    else:
        raise ValueError(
            f"Unknown mode for contacts: {mode}. "
            "Supported modes are 'CG' and 'AA'."
        )

    def contact_energy(r, sigma):
        return epsilon * (
            COEFF_REPULSION * (sigma / r) ** POW_REPULSION
            - COEFF_ATTRACTION * (sigma / r) ** POW_ATTRACTION
        )

    def isoenergetic_distance(r1, r2):
        return (
            COEFF_ATTRACTION
            / COEFF_REPULSION
            * (
                (1 / r1**POW_ATTRACTION - 1 / r2**POW_ATTRACTION)
                / (1 / r1**POW_REPULSION - 1 / r2**POW_REPULSION)
            )
        ) ** (1 / (POW_REPULSION - POW_ATTRACTION))

    reference_root = reference_xml.getroot()
    if reference_root is None:
        raise ValueError(
            "The reference XML tree is empty or malformed."
        )
    additional_root = additional_xml.getroot()
    if additional_root is None:
        raise ValueError(
            "The additional XML tree is empty or malformed."
        )
    multibasin_root = multibasin_xml.getroot()
    if multibasin_root is None:
        raise ValueError(
            "The multibasin XML tree is empty or malformed."
        )

    contacts_converted_to_reference = {
        "i": [],
        "j": [],
        "A": [],
        "B": [],
    }

    for contact_type in additional_root.findall(
        ".//contacts/contacts_type"
    ):
        for interaction in contact_type.findall("interaction"):
            # Convert contact indices from additional to reference

            ii = int(interaction.attrib["i"])
            jj = int(interaction.attrib["j"])

            i = idx_from_additional_to_reference.get(ii)
            j = idx_from_additional_to_reference.get(jj)

            # when going thru additional, there's no option of
            # sigma_reference not existing if sigma_additional exists
            if i is not None and j is not None:
                contacts_converted_to_reference["i"].append(min(i, j))
                contacts_converted_to_reference["j"].append(max(i, j))

                contacts_converted_to_reference["A"].append(
                    interaction.attrib["A"]
                )
                contacts_converted_to_reference["B"].append(
                    interaction.attrib["B"]
                )

    i_list = contacts_converted_to_reference["i"]
    j_list = contacts_converted_to_reference["j"]
    contacts_converted_to_reference["sigma_reference"] = (
        np.linalg.norm(
            reference_pdb.loc[i_list, ["x", "y", "z"]].values
            - reference_pdb.loc[j_list, ["x", "y", "z"]].values,
            axis=1,
        )
        * 0.1  # Convert to nm
    ) * scale_distances

    df_additional = pandas.DataFrame(
        data=contacts_converted_to_reference
    )
    df_additional[["A", "B"]] = df_additional[["A", "B"]].astype(
        float
    )
    df_additional[["i", "j"]] = df_additional[["i", "j"]].astype(int)
    df_additional["sigma_additional"] = (
        df_additional["A"]
        / df_additional["B"]
        * COEFF_ATTRACTION
        / COEFF_REPULSION
    ) ** (1 / (POW_REPULSION - POW_ATTRACTION))

    df_additional.drop(["A", "B"], axis=1, inplace=True)

    contacts_in_reference = {
        "i": [],
        "j": [],
        "A": [],
        "B": [],
    }

    idx_from_reference_to_additional = {
        v: k for k, v in idx_from_additional_to_reference.items()
    }

    for contacts_type in reference_root.findall(
        ".//contacts/contacts_type"
    ):
        for interaction in contacts_type.findall("interaction"):

            i = int(interaction.attrib["i"])
            j = int(interaction.attrib["j"])

            contacts_in_reference["i"].append(min(i, j))
            contacts_in_reference["j"].append(max(i, j))
            contacts_in_reference["A"].append(interaction.attrib["A"])
            contacts_in_reference["B"].append(interaction.attrib["B"])

    # convert index to float to allow NaN and add NaN row
    nan_row = pandas.DataFrame([{}], index=[np.nan])
    float_indexed_additional_pdb = pandas.concat([additional_pdb, nan_row])

    ii_list = np.array(
        [
            idx_from_reference_to_additional.get(i)
            for i in contacts_in_reference["i"]
        ],
        dtype=float,
    )
    jj_list = np.array(
        [
            idx_from_reference_to_additional.get(j)
            for j in contacts_in_reference["j"]
        ],
        dtype=float,
    )

    # this will give NaN for the contacts that are not in the additional PDB
    contacts_in_reference["sigma_additional"] = (
        np.linalg.norm(
            float_indexed_additional_pdb.loc[ii_list, ["x", "y", "z"]].values
            - float_indexed_additional_pdb.loc[jj_list, ["x", "y", "z"]].values,
            axis=1,
        )
        * 0.1  # Convert to nm
    ) * scale_distances

    df_reference = pandas.DataFrame(data=contacts_in_reference)
    df_reference[["A", "B"]] = df_reference[["A", "B"]].astype(float)
    df_reference[["i", "j"]] = df_reference[["i", "j"]].astype(int)
    df_reference["sigma_reference"] = (
        df_reference["A"]
        / df_reference["B"]
        * COEFF_ATTRACTION
        / COEFF_REPULSION
    ) ** (1 / (POW_REPULSION - POW_ATTRACTION))

    # pair is not in the additional PDB, use reference distance
    df_reference["sigma_additional"] = df_reference[
        "sigma_additional"
    ].fillna(df_reference["sigma_reference"])

    df_reference["epsilon"] = (
        df_reference["A"]
        / COEFF_REPULSION
        * df_reference["sigma_reference"] ** -POW_REPULSION
    )

    df_reference.drop(["A", "B"], axis=1, inplace=True)

    if mode == "CG" and not np.isclose(
        np.nanmean(df_reference["epsilon"]), epsilon, atol=0.01
    ):
        print(
            "Warning! Contacts do not have the expected epsilon value for a CG model."
            f"Expected: {epsilon}, Found: {np.nanmean(df_reference['epsilon'])}."
        )

    # Merge the two DataFrames on i, j
    merged_contacts = pandas.merge(
        df_reference,
        df_additional,
        on=["i", "j"],
        how="outer",
        suffixes=("_1", "_2"),
        indicator="source",
    )  # this identifies common contact pairs

    if (
        np.mean(
            np.isclose(
                merged_contacts.loc[
                    merged_contacts.source == "both",
                    "sigma_reference_1",
                ],
                merged_contacts.loc[
                    merged_contacts.source == "both",
                    "sigma_reference_2",
                ],
                atol=0.01,
            )
        )
        == 1.0
    ):
        # prioritize the distance from the PDB
        merged_contacts["sigma_reference"] = merged_contacts[
            "sigma_reference_2"
        ].combine_first(merged_contacts["sigma_reference_1"])

    else:
        raise ValueError(
            "Inconsistent values of sigma_reference in the reference and additional XML files."
        )

    if (
        np.mean(
            np.isclose(
                merged_contacts.loc[
                    merged_contacts.source == "both",
                    "sigma_additional_1",
                ],
                merged_contacts.loc[
                    merged_contacts.source == "both",
                    "sigma_additional_2",
                ],
                atol=0.01,
            )
        )
        == 1.0
    ):
        # prioritize the distance from the PDB
        merged_contacts["sigma_additional"] = merged_contacts[
            "sigma_additional_1"
        ].combine_first(merged_contacts["sigma_additional_2"])

    else:
        raise ValueError(
            "Inconsistent values of sigma_additional in the reference and additional XML files."
        )

    assert (
        merged_contacts["sigma_reference"].notna().all()
    ), "NaN found in sigma_reference."
    assert (
        merged_contacts["sigma_additional"].notna().all()
    ), "NaN found in sigma_additional."

    # common contacts

    where_r1_r2_equal = (
        merged_contacts["sigma_reference"]
        == merged_contacts["sigma_additional"]
    )

    merged_contacts.loc[where_r1_r2_equal, "sigma_iso"] = (
        merged_contacts.loc[where_r1_r2_equal, "sigma_reference"]
    )
    merged_contacts.loc[~where_r1_r2_equal, "sigma_iso"] = (
        isoenergetic_distance(
            merged_contacts.loc[
                ~where_r1_r2_equal, "sigma_reference"
            ],
            merged_contacts.loc[
                ~where_r1_r2_equal, "sigma_additional"
            ],
        )
    )

    merged_contacts["epsilon_iso"] = np.abs(
        contact_energy(
            r=merged_contacts["sigma_reference"],
            sigma=merged_contacts["sigma_iso"],
        )
    )

    ## for the cases where the eps_iso/eps <= 0.5
    # when both contacts are called by Shadow, go back to the lower distance
    merged_contacts.loc[
        (merged_contacts["epsilon_iso"] <= 0.5 * epsilon)
        & (merged_contacts["source"] == "both"),
        "sigma_iso",
    ] = np.min(
        merged_contacts.loc[
            (merged_contacts["epsilon_iso"] <= 0.5 * epsilon)
            & (merged_contacts["source"] == "both"),
            ["sigma_reference", "sigma_additional"],
        ].values,
        axis=1,
    )
    # when only one of the contacts is called by Shadow
    # go back to that one
    merged_contacts.loc[
        (merged_contacts["epsilon_iso"] <= 0.5 * epsilon)
        & (merged_contacts["source"] == "left_only"),
        "sigma_iso",
    ] = merged_contacts.loc[
        (merged_contacts["epsilon_iso"] <= 0.5 * epsilon)
        & (merged_contacts["source"] == "left_only"),
        "sigma_reference",
    ].values
    merged_contacts.loc[
        (merged_contacts["epsilon_iso"] <= 0.5 * epsilon)
        & (merged_contacts["source"] == "right_only"),
        "sigma_iso",
    ] = merged_contacts.loc[
        (merged_contacts["epsilon_iso"] <= 0.5 * epsilon)
        & (merged_contacts["source"] == "right_only"),
        "sigma_additional",
    ].values

    # fix source assignment
    merged_contacts.loc[
        (
            (
                merged_contacts["sigma_iso"]
                != merged_contacts["sigma_reference"]
            )
            & (
                merged_contacts["sigma_iso"]
                != merged_contacts["sigma_additional"]
            )
        ),
        "source",
    ] = "both"

    merged_contacts.loc[
        merged_contacts["sigma_iso"]
        == merged_contacts["sigma_reference"],
        "source",
    ] = "left_only"

    merged_contacts.loc[
        (
            (
                merged_contacts["sigma_iso"]
                != merged_contacts["sigma_reference"]
            )
            & (
                merged_contacts["sigma_iso"]
                == merged_contacts["sigma_additional"]
            )
        ),
        "source",
    ] = "right_only"

    if mode == "AA":
        print(
            "\n\tAdjusting epsilon_c and dihedral weights for AA model...",
            end=" ",
            flush=True,
        )

        # adjust epsilon and the weight of the dihedrals
        num_common_contacts = len(
            merged_contacts.loc[merged_contacts["source"] == "both"]
        )

        dihedrals, df_dihedrals = _count_dihedrals(
            reference_pdb, multibasin_xml
        )
        num_backbone_dihedrals = (
            dihedrals["protein_bb"] + dihedrals["nucleic_bb"]
        )

        Nc = num_common_contacts
        Nbb = num_backbone_dihedrals
        Nscp = dihedrals["protein_sc"]
        Nscn = dihedrals["nucleic_sc"]

        N = len(reference_pdb)

        # variables: [Ec, Ebb, Escp, Escn]
        A = [
            [0, 1, 0, -1],
            [0, 1, -2, 0],
            [Nc, 2 * Nbb, -2 * Nscp, -2 * Nscn],
            [Nc, Nbb, Nscp, Nscn],
        ]
        b = [0, 0, 0, N]
        solution = np.linalg.solve(A, b)
        epsilon, weight_bb, weight_scp, weight_scn = solution

        df_dihedrals["epsilon"] = weight_bb
        df_dihedrals.loc[
            (df_dihedrals["molecule_type"] == "protein")
            & (df_dihedrals["dihedral_type"] == "sc"),
            "epsilon",
        ] = weight_scp
        df_dihedrals.loc[
            (df_dihedrals["molecule_type"] == "nucleic")
            & (df_dihedrals["dihedral_type"] == "sc"),
            "epsilon",
        ] = weight_scn

        multibasin_xml = _update_dihedral_weights(
            multibasin_xml,
            df_dihedrals,
        )
        print("Done.", flush=True)

    # calculate the new A and B parameters
    merged_contacts["A"] = (
        epsilon
        * COEFF_REPULSION
        * merged_contacts["sigma_iso"] ** POW_REPULSION
    )
    merged_contacts["B"] = (
        epsilon
        * COEFF_ATTRACTION
        * merged_contacts["sigma_iso"] ** POW_ATTRACTION
    )

    ## replace new contacts in the mutlibasin_xml
    contact_information = {
        "i": [],
        "j": [],
        "distance": [],
        "source": [],
    }

    for contact_type in multibasin_root.findall(
        ".//contacts/contacts_type"
    ):
        # 1) Build a new list of just the non-interaction children
        edited_contacts = [
            element
            for element in contact_type
            if element.tag != "interaction"
        ]

        # 2) Append your merged interactions
        for row in merged_contacts.itertuples(index=False):
            edited_contacts.append(
                ET.Element(
                    "interaction",
                    attrib={
                        "i": str(row.i),
                        "j": str(row.j),
                        "A": f"{row.A:.5e}",
                        "B": f"{row.B:.5e}",
                    },
                )
            )
            contact_information["i"].append(row.i)
            contact_information["j"].append(row.j)
            contact_information["distance"].append(row.sigma_iso)
            if row.source == "left_only":
                contact_information["source"].append("left")
            elif row.source == "right_only":
                contact_information["source"].append("right")
            else:
                contact_information["source"].append("common")

        # 3) In one go, reassign the children of dihedral_type
        contact_type[:] = edited_contacts

    return multibasin_xml, pandas.DataFrame(contact_information)


def _update_exclusions(
    xml: ET.ElementTree,
    top: dict[str, pandas.DataFrame],
) -> dict[str, pandas.DataFrame]:
    """
    Updates the exclusions in the topology based on the contacts defined in the XML.

    Args:
        xml (ET.ElementTree): The XML tree containing contact information.
        top (dict[str, pandas.DataFrame]): The topology dictionary containing atom information.

    Returns:
        dict[str, pandas.DataFrame]: The updated topology with updated exclusions.
    """
    root = xml.getroot()
    if root is None:
        raise ValueError("The XML tree is empty or malformed.")

    exclusions = {
        "ai": [],
        "aj": [],
    }

    for contact_type in root.findall(".//contacts/contacts_type"):
        for interaction in contact_type.findall("interaction"):
            i = int(interaction.attrib["i"])
            j = int(interaction.attrib["j"])

            exclusions["ai"].append(i)
            exclusions["aj"].append(j)

    top["exclusions"] = pandas.DataFrame(exclusions)

    return top


def scale_contacts(
    xml: ET.ElementTree,
    atom_pairs: np.ndarray,
    scale_by: float | None = None,
    scale_to: float | None = None,
) -> ET.ElementTree:
    """
    Scales the contact strength in the XML file by a given scale factor.

    Args:
        xml (ET.ElementTree): The XML tree containing contact information.
        atom_pairs (np.ndarray): Array of atom pairs for which contacts are defined.
        scale_by (float): Factor by which to scale the contact distances.

    Returns:
        ET.ElementTree: The modified XML tree with scaled contacts.
    """
    if scale_by is not None and scale_to is not None:
        raise ValueError(
            "Only one of `scale_by` or `scale_to` should be provided."
        )
    if scale_to is not None and scale_by is None:
        if scale_to <= 0:
            raise ValueError("`scale_to` must be greater than zero.")
        scale = lambda x: scale_to
    elif scale_to is None and scale_by is not None:
        scale = lambda x: scale_by * float(x)
    else:
        raise ValueError(
            "Either `scale_by` or `scale_to` must be provided."
        )

    edited_xml = copy.deepcopy(xml)

    root = edited_xml.getroot()
    if root is None:
        raise ValueError("The XML tree is empty or malformed.")

    if scale_by == 0:
        raise ValueError(
            "Scale factor must be greater than zero. Use `delete_contacts` instead to remove contacts."
        )

    contacts_to_scale = set(
        tuple(sorted(pair)) for pair in atom_pairs
    )

    for contact_type in root.findall(".//contacts/contacts_type"):
        for interaction in contact_type.findall("interaction"):
            i = int(interaction.attrib["i"])
            j = int(interaction.attrib["j"])

            if tuple(sorted([i, j])) in contacts_to_scale:
                interaction.attrib["A"] = (
                    f"{scale(interaction.attrib['A']):.5e}"
                )
                interaction.attrib["B"] = (
                    f"{scale(interaction.attrib['B']):.5e}"
                )

    return edited_xml


def delete_contacts(
    xml: ET.ElementTree,
    top: dict[str, pandas.DataFrame],
    atom_pairs: np.ndarray,
) -> tuple[ET.ElementTree, dict[str, pandas.DataFrame]]:
    """
    Deletes contacts in the XML file for specified atom pairs.

    Args:
        xml (ET.ElementTree): The XML tree containing contact information.
        atom_pairs (np.ndarray): A 2D NumPy array of integer pairs, where each pair represents the indices of atoms whose contacts should be deleted.

    Returns:
        ET.ElementTree: The modified XML tree with specified contacts removed.
    """
    edited_xml = copy.deepcopy(xml)
    edited_top = copy.deepcopy(top)

    root = edited_xml.getroot()
    if root is None:
        raise ValueError("The XML tree is empty or malformed.")

    set_contacts_to_delete = set(
        tuple(sorted(pair)) for pair in atom_pairs
    )

    for contact_type in list(
        root.findall(".//contacts/contacts_type")
    ):
        elements_to_keep = []
        for element in contact_type:
            if element.tag != "interaction":
                elements_to_keep.append(element)
            elif element.tag == "interaction":

                i = int(element.attrib["i"])
                j = int(element.attrib["j"])

                if (
                    tuple(sorted([i, j]))
                    not in set_contacts_to_delete
                ):
                    elements_to_keep.append(element)
            else:
                elements_to_keep.append(element)

        contact_type[:] = elements_to_keep

    edited_top = _update_exclusions(
        xml=edited_xml,
        top=edited_top,
    )

    return edited_xml, edited_top


def load_force_field(
    pdb_file: str = "smog.pdb",
    top_file: str = "smog.top",
    xml_file: str = "smog.xml",
    coarse_grain: bool = True,
    ignore_HETATM: bool = True,
) -> tuple[
    pandas.DataFrame,
    dict[str, pandas.DataFrame],
    ET.ElementTree,
]:
    pdb = pdb_tools.read_pdb(
        pdb_file,
        coarse_grain=coarse_grain,
        ignore_HETATM=ignore_HETATM,
    )
    top = read_top(top_file)
    xml = ET.parse(xml_file)

    return pdb, top, xml


def define_multibasin_model(
    reference_top: dict[str, pandas.DataFrame],
    reference_xml: ET.ElementTree,
    reference_pdb: pandas.DataFrame,
    additional_top: dict[str, pandas.DataFrame],
    additional_xml: ET.ElementTree,
    additional_pdb: pandas.DataFrame,
    idx_from_additional_to_reference: dict[int, int],
    mode_angles: str = "middle",
    mode_contacts: str = "CG",
    xml_file: str = "smog.xml",
    top_file: str = "smog.top",
    contact_file: str = "contacts.pkl",
    scale_contact_distances: float = 1.0,
) -> tuple[
    dict[str, pandas.DataFrame], ET.ElementTree, pandas.DataFrame
]:
    """
    Updates the dihedrals and angles from the reference topology to the middle value between the two topologies.
    """

    multibasin_top = copy.deepcopy(reference_top)
    multibasin_xml = copy.deepcopy(reference_xml)

    # For the bonds
    print("Processing bonds...", flush=True, end="")
    multibasin_top, multibasin_xml = _process_bonds(
        reference_top,
        additional_top,
        multibasin_top,
        idx_from_additional_to_reference,
        multibasin_xml,
    )
    print(" Done.", flush=True)

    # For the angles

    print("Processing angles...", flush=True, end="")
    multibasin_top, multibasin_xml = _process_angles(
        reference_top,
        additional_top,
        multibasin_top,
        idx_from_additional_to_reference,
        multibasin_xml,
        mode_angles,
    )
    print(" Done.", flush=True)

    # For the dihedrals

    print("Processing dihedrals...", flush=True, end="")
    multibasin_xml = _process_dihedrals(
        reference_xml,
        additional_xml,
        multibasin_xml,
        idx_from_additional_to_reference,
    )
    print(" Done.", flush=True)

    # For the contacts

    print("Processing contacts...", flush=True, end="")
    multibasin_xml, df_contacts = _process_contacts(
        reference_xml,
        reference_pdb,
        additional_xml,
        additional_pdb,
        multibasin_xml,
        idx_from_additional_to_reference,
        mode_contacts,
        scale_contact_distances,
    )

    multibasin_top = _update_exclusions(
        xml=multibasin_xml,
        top=multibasin_top,
    )
    print(" Done.", flush=True)

    # Save the new files

    print("Saving files...", flush=True, end="")

    save_top(multibasin_top, top_file)
    save_xml(multibasin_xml, xml_file)

    df_contacts.to_pickle(contact_file)

    print(" Done.", flush=True)

    return multibasin_top, multibasin_xml, df_contacts


def xml2contacts(
    xml: ET.ElementTree,
    filename: str = "smog.CG.contacts.CG",
) -> None:
    """
    Extract contacts from XML and write contact file.

    Args:
        xml (ET.ElementTree): The XML tree containing contact information.
        filename (str): The name of the output file to save contacts.
    """

    root = xml.getroot()
    if root is None:
        raise ValueError("The XML tree is empty or malformed.")

    lines = []

    for contact_type in root.findall(".//contacts/contacts_type"):
        for interaction in contact_type.findall("interaction"):
            i = int(interaction.attrib["i"])
            j = int(interaction.attrib["j"])

            lines.append(f"1 {i} 1 {j}\n")

    with open(filename, "w") as f:
        f.writelines(lines)


def remove_unstable_dihedrals(
    xml: ET.ElementTree,
    top: dict[str, pandas.DataFrame] | None = None,
) -> ET.ElementTree:
    """
    Remove dihedral interactions that may cause numerical instability.
    When the angles in a dihedral are close to 0 or 180 degrees,
    the dihedral potential can become very steep, leading to
    numerical instability during simulations.

    Args:
        xml (ET.ElementTree): The XML tree containing dihedral information.
        top (dict[str, pandas.DataFrame] | None): The topology dictionary containing the angle information, if that is not defined in the xml.
    Returns:
        ET.ElementTree: The modified XML tree with specified dihedrals removed.
    """

    def _strlist2tuple(*strlist: list) -> tuple:
        return tuple(sorted(map(int, strlist)))

    edited_xml = copy.deepcopy(xml)
    root = edited_xml.getroot()

    idx_in_faulty_angles = set()

    for angle_type in root.findall(".//angles/angles_type"):
        for interaction in angle_type.findall("interaction"):
            if (
                float(interaction.attrib["theta1"]) <= np.deg2rad(30)
                or float(interaction.attrib["theta1"])
                >= np.deg2rad(150)
            ) or (
                float(interaction.attrib["theta2"]) <= np.deg2rad(30)
                or float(interaction.attrib["theta2"])
                >= np.deg2rad(150)
            ):
                idx_in_faulty_angles.add(
                    _strlist2tuple(
                        interaction.attrib["i"],
                        interaction.attrib["j"],
                        interaction.attrib["k"],
                    )
                )

    if (
        top is not None
        and "angles" in top
        and "th0(deg)" in top["angles"]
    ):
        for row in top["angles"].itertuples():
            if (row["th0(deg)"] <= 30) or (row["th0(deg)"] >= 150):
                idx_in_faulty_angles.add(
                    _strlist2tuple(
                        row.ai,
                        row.aj,
                        row.ak,
                    )
                )

    for dihedral_type in root.findall(".//dihedrals/dihedrals_type"):
        modified_dihedrals = []
        for element in dihedral_type:
            if element.tag == "interaction":
                dihedral_idx = set()

                dihedral_idx.add(
                    _strlist2tuple(
                        element.attrib["j"],
                        element.attrib["k"],
                        element.attrib["l"],
                    )
                )
                dihedral_idx.add(
                    _strlist2tuple(
                        element.attrib["i"],
                        element.attrib["k"],
                        element.attrib["l"],
                    )
                )
                dihedral_idx.add(
                    _strlist2tuple(
                        element.attrib["i"],
                        element.attrib["j"],
                        element.attrib["l"],
                    )
                )
                dihedral_idx.add(
                    _strlist2tuple(
                        element.attrib["i"],
                        element.attrib["j"],
                        element.attrib["k"],
                    )
                )

                if (
                    len(
                        dihedral_idx.intersection(
                            idx_in_faulty_angles
                        )
                    )
                    > 0
                ):
                    continue
                else:
                    modified_dihedrals.append(element)
            else:
                modified_dihedrals.append(element)
        dihedral_type[:] = modified_dihedrals

    return edited_xml
