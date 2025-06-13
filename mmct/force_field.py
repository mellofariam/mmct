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
            bonds_converted_to_reference["ai"].append(i)
            bonds_converted_to_reference["aj"].append(j)
            bonds_converted_to_reference["func"].append(row["func"])
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
    df_bonds_additional[["r0(nm)", "Kb"]] = df_bonds_additional[
        ["r0(nm)", "Kb"]
    ].astype(float)

    merged_bonds = pandas.merge(
        reference_top["bonds"],
        df_bonds_additional,
        left_on=["ai", "aj", "func", "Kb"],
        right_on=["ai", "aj", "func", "Kb"],
        how="outer",
        suffixes=("_1", "_2"),
        indicator=True,
    )

    merged_bonds.loc[
        merged_bonds["_merge"] == "left_only", "r0(nm)_2"
    ] = merged_bonds.loc[
        merged_bonds["_merge"] == "left_only", "r0(nm)_1"
    ]
    merged_bonds.loc[
        merged_bonds["_merge"] == "right_only", "r0(nm)_1"
    ] = merged_bonds.loc[
        merged_bonds["_merge"] == "right_only", "r0(nm)_2"
    ]

    merged_bonds["r0(nm)"] = np.nanmean(
        merged_bonds[["r0(nm)_1", "r0(nm)_2"]],
        axis=1,
    )

    multibasin_top["bonds"] = merged_bonds[
        ["ai", "aj", "func", "r0(nm)", "Kb"]
    ].reset_index(drop=True)

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
    additional_root = additional_xml.getroot()
    multibasin_root = multibasin_xml.getroot()

    dihedrals_converted_to_reference = {
        "i": [],
        "j": [],
        "k": [],
        "l": [],
        "theta0": [],
        "weight": [],
        "multiplicity": [],
    }

    for dihedral_type in additional_root.findall(
        ".//dihedrals/dihedrals_type"
    ):
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
                dihedrals_converted_to_reference["i"].append(i)
                dihedrals_converted_to_reference["j"].append(j)
                dihedrals_converted_to_reference["k"].append(k)
                dihedrals_converted_to_reference["l"].append(l)
                dihedrals_converted_to_reference["theta0"].append(
                    interaction.attrib["theta0"]
                )
                dihedrals_converted_to_reference["weight"].append(
                    interaction.attrib["weight"]
                )
                dihedrals_converted_to_reference[
                    "multiplicity"
                ].append(interaction.attrib["multiplicity"])

    df_additional = pandas.DataFrame(
        data=dihedrals_converted_to_reference
    )
    df_additional[["theta0", "weight"]] = df_additional[
        ["theta0", "weight"]
    ].astype(float)
    df_additional[["i", "j", "k", "l", "multiplicity"]] = (
        df_additional[["i", "j", "k", "l", "multiplicity"]].astype(
            int
        )
    )

    dihedrals_in_reference = {
        "i": [],
        "j": [],
        "k": [],
        "l": [],
        "theta0": [],
        "weight": [],
        "multiplicity": [],
    }
    for dihedral_type in reference_root.findall(
        ".//dihedrals/dihedrals_type"
    ):
        for interaction in dihedral_type.findall("interaction"):
            i = int(interaction.attrib["i"])
            j = int(interaction.attrib["j"])
            k = int(interaction.attrib["k"])
            l = int(interaction.attrib["l"])

            dihedrals_in_reference["i"].append(i)
            dihedrals_in_reference["j"].append(j)
            dihedrals_in_reference["k"].append(k)
            dihedrals_in_reference["l"].append(l)
            dihedrals_in_reference["theta0"].append(
                interaction.attrib["theta0"]
            )
            dihedrals_in_reference["weight"].append(
                interaction.attrib["weight"]
            )
            dihedrals_in_reference["multiplicity"].append(
                interaction.attrib["multiplicity"]
            )

    df_reference = pandas.DataFrame(data=dihedrals_in_reference)
    df_reference[["theta0", "weight"]] = df_reference[
        ["theta0", "weight"]
    ].astype(float)
    df_reference[
        [
            "i",
            "j",
            "k",
            "l",
            "multiplicity",
        ]
    ] = df_reference[
        [
            "i",
            "j",
            "k",
            "l",
            "multiplicity",
        ]
    ].astype(
        int
    )

    # Merge the two DataFrames on i, j, k, l

    merged_dihedrals = pandas.merge(
        df_reference,
        df_additional,
        on=["i", "j", "k", "l", "weight", "multiplicity"],
        how="outer",
        suffixes=("_1", "_2"),
        indicator=True,
    )

    merged_dihedrals.loc[
        merged_dihedrals["_merge"] == "left_only", "theta0_2"
    ] = merged_dihedrals.loc[
        merged_dihedrals["_merge"] == "left_only", "theta0_1"
    ]
    merged_dihedrals.loc[
        merged_dihedrals["_merge"] == "right_only", "theta0_1"
    ] = merged_dihedrals.loc[
        merged_dihedrals["_merge"] == "right_only", "theta0_2"
    ]

    merged_dihedrals["theta0"] = _angular_midpoint(
        merged_dihedrals["theta0_1"].values,
        merged_dihedrals["theta0_2"].values,
    )

    ## replace new dihedrals in the mutlibasin_xml

    for dihedral_type in multibasin_root.findall(
        ".//dihedrals/dihedrals_type"
    ):
        # 1) Build a new list of just the non-interaction children
        edited_dihedrals = [
            element
            for element in dihedral_type
            if element.tag != "interaction"
        ]

        # 2) Append your merged interactions
        for row in merged_dihedrals.itertuples(index=False):
            edited_dihedrals.append(
                ET.Element(
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
                        "multiplicity": str(row.multiplicity),
                    },
                )
            )

        # 3) In one go, reassign the children of dihedral_type
        dihedral_type[:] = edited_dihedrals

    return multibasin_xml


def _process_contacts(
    reference_xml: ET.ElementTree,
    additional_xml: ET.ElementTree,
    multibasin_xml: ET.ElementTree,
    idx_from_additional_to_reference: dict[int, int],
) -> ET.ElementTree:
    """
    Processes contacts from the reference and additional XML files,
    converting them to the reference indices and merging them.
    The final contacts are averages of the parameters in both structures.
    """

    reference_root = reference_xml.getroot()
    additional_root = additional_xml.getroot()
    multibasin_root = multibasin_xml.getroot()

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
            # Convert dihedral indices from additional to reference
            i = idx_from_additional_to_reference.get(
                int(interaction.attrib["i"])
            )
            j = idx_from_additional_to_reference.get(
                int(interaction.attrib["j"])
            )

            if i is not None and j is not None:
                contacts_converted_to_reference["i"].append(i)
                contacts_converted_to_reference["j"].append(j)

                contacts_converted_to_reference["A"].append(
                    interaction.attrib["A"]
                )
                contacts_converted_to_reference["B"].append(
                    interaction.attrib["B"]
                )

    df_additional = pandas.DataFrame(
        data=contacts_converted_to_reference
    )
    df_additional[["A", "B"]] = df_additional[["A", "B"]].astype(
        float
    )
    df_additional[["i", "j"]] = df_additional[["i", "j"]].astype(int)

    contacts_in_reference = {"i": [], "j": [], "A": [], "B": []}
    for dihedral_type in reference_root.findall(
        ".//contacts/contacts_type"
    ):
        for interaction in dihedral_type.findall("interaction"):
            contacts_in_reference["i"].append(interaction.attrib["i"])
            contacts_in_reference["j"].append(interaction.attrib["j"])
            contacts_in_reference["A"].append(interaction.attrib["A"])
            contacts_in_reference["B"].append(interaction.attrib["B"])

    df_reference = pandas.DataFrame(data=contacts_in_reference)
    df_reference[["A", "B"]] = df_reference[["A", "B"]].astype(float)
    df_reference[["i", "j"]] = df_reference[["i", "j"]].astype(int)

    # Merge the two DataFrames on i, j

    merged_contacts = pandas.merge(
        df_reference,
        df_additional,
        on=["i", "j"],
        how="outer",
        suffixes=("_1", "_2"),
        indicator=True,
    )

    merged_contacts.loc[
        merged_contacts["_merge"] == "left_only", "theta0_2"
    ] = merged_contacts.loc[
        merged_contacts["_merge"] == "left_only", "theta0_1"
    ]
    merged_contacts.loc[
        merged_contacts["_merge"] == "right_only", "theta0_1"
    ] = merged_contacts.loc[
        merged_contacts["_merge"] == "right_only", "theta0_2"
    ]

    merged_contacts["theta0"] = _angular_midpoint(
        merged_contacts["theta0_1"].values,
        merged_contacts["theta0_2"].values,
    )

    ## replace new contacts in the mutlibasin_xml

    for dihedral_type in multibasin_root.findall(
        ".//contacts/contacts_type"
    ):
        # 1) Build a new list of just the non-interaction children
        edited_contacts = [
            element
            for element in dihedral_type
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
                        "k": str(row.k),
                        "l": str(row.l),
                        "theta0": f"{row.theta0:.5e}",
                        "weight": (
                            f"{row.weight:.5e}"
                            if row.weight != 1
                            else "1"
                        ),
                        "multiplicity": str(row.multiplicity),
                    },
                )
            )

        # 3) In one go, reassign the children of dihedral_type
        dihedral_type[:] = edited_contacts

    return multibasin_xml


def define_multibasin_model(
    reference_top: dict[str, pandas.DataFrame],
    reference_xml: str,
    additional_top: dict[str, pandas.DataFrame],
    additional_xml: str,
    idx_from_reference_to_additional: dict[int, int],
    idx_from_additional_to_reference: dict[int, int],
    mode_angles: str = "middle",
    xmlfile: str = "smog.xml",
    topfile: str = "smog.top",
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

    multibasin_xml = copy.deepcopy(reference_xml)

    # For the angles

    multibasin_top, multibasin_xml = _process_angles(
        reference_top,
        additional_top,
        multibasin_top,
        idx_from_additional_to_reference,
        multibasin_xml,
        mode_angles,
    )

    # For the bonds

    multibasin_top, multibasin_xml = _process_bonds(
        reference_top,
        additional_top,
        multibasin_top,
        idx_from_additional_to_reference,
        multibasin_xml,
    )

    # For the dihedrals

    multibasin_xml = _process_dihedrals(
        reference_xml,
        additional_xml,
        multibasin_xml,
        idx_from_additional_to_reference,
    )

    # For the contacts

    multibasin_xml = _process_contacts(
        reference_xml,
        additional_xml,
        multibasin_xml,
        idx_from_additional_to_reference,
    )

    save_top(multibasin_top, topfile)
    save_xml(multibasin_xml, xmlfile)

    return multibasin_top, multibasin_xml
