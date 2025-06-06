import pandas


def extract_CG_data(pdb_path):
    data = {
        "index": [],
        "record": [],
        "atom": [],
        "residue": [],
        "residue_number": [],
        "chain_id": [],
        "x": [],
        "y": [],
        "z": [],
    }

    with open(pdb_path, "r") as pdb_file:

        idx = 1
        for line in pdb_file:
            record = (
                line[0:6].replace(" ", "")
                if line[3] != "\n"
                else line[0:3]
            )

            if record in ["ATOM", "HETATM"]:

                atom_name = line[12:16].replace(" ", "")
                residue = line[17:21].replace(" ", "")
                chain = line[21].replace(" ", "")
                residue_number = int(line[22:26].replace(" ", ""))

                if (
                    atom_name == "CA"
                    or (
                        atom_name == "P"
                        and residue
                        in [
                            "A",
                            "C",
                            "G",
                            "U",
                            "AM",
                            "CM",
                            "GM",
                            "UM",
                        ]
                    )
                    or (
                        atom_name == "O5*"
                        and residue
                        in [
                            "A0P",
                            "C0P",
                            "G0P",
                            "U0P",
                            "A0PM",
                            "C0PM",
                            "G0PM",
                            "U0PM",
                        ]
                    )
                ):
                    data["record"].append(record)
                    data["index"].append(idx)
                    data["atom"].append(atom_name)
                    data["residue"].append(residue)
                    data["chain_id"].append(chain)
                    data["residue_number"].append(residue_number)

                    data["x"].append(float(line[30:38]))
                    data["y"].append(float(line[38:46]))
                    data["z"].append(float(line[46:54]))

                    idx += 1

    df = pandas.DataFrame(data)

    return df


def write_seqres(chains, complex, output_file):
    """
    Write an RNA sequence in the SEQRES format for a PDB file.

    Parameters:
    chains (list): List of chains.
    complex (mol.Complex): Complex object.
    output_file (str): Path to the output PDB file.
    """
    with open(output_file, "w") as f:
        seqres_lines = []
        for chain_id in chains:
            sequence = complex.molecules[
                chain_id
            ].reference_sequence_as_list
            for i in range(0, len(sequence), 13):
                segment = sequence[i : i + 13]
                seqres_line = (
                    f"SEQRES {i//13 + 1:>3} {chain_id}  {len(sequence):>3}    "
                    + "   ".join(segment)
                )
                seqres_lines.append(seqres_line)

        f.writelines("\n".join(seqres_lines))


def save_pdb(
    df,
    filename=None,
    ignore_HETATM=False,
    file_mode="w",
    add_TER=True,
    add_END=True,
):
    pdb_string_formats = {
        "coordinates": "{record:6s}{index:5d} {atom:4s} {residue:4s}{chain_id:1s}{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f} {occupancy:6.2f}{b_factor:6.2f}          {element:>2s}  \n",
        "ter": "TER   {index:5d}      {residue:3s} {chain_id:1s}{residue_number:4d}\n",
        "end": "END\n",
    }

    with open(filename, file_mode) as pdb_file:
        index = 1
        previous_chain_id = None
        for _, row in df.iterrows():
            if row.record == "ATOM":
                pdb_file.write(
                    pdb_string_formats["coordinates"].format(
                        record="ATOM",
                        index=index,
                        atom=row.atom,
                        residue=row.residue,
                        chain_id=row.chain_id,
                        residue_number=int(row.residue_number),
                        x=float(row.x),
                        y=float(row.y),
                        z=float(row.z),
                        occupancy=float(1),
                        b_factor=float(0),
                        element=str(row.atom)[0],
                    )
                )
                index += 1
            elif row.record == "HETATM" and not ignore_HETATM:
                pdb_file.write(
                    pdb_string_formats["coordinates"].format(
                        record=row.record,
                        index=index,
                        atom=row.atom,
                        residue=row.residue,
                        chain_id=row.chain_id,
                        residue_number=int(row.residue_number),
                        x=float(row.x),
                        y=float(row.y),
                        z=float(row.z),
                        occupancy=float(row.occupancy),
                        b_factor=float(row.b_factor),
                        element=row.element,
                    )
                )
                index += 1

            # Add TER records for the last residue in each chain if required
            if (
                add_TER
                and previous_chain_id != row.chain_id
                and previous_chain_id is not None
            ):
                # Only add TER if the chain has changed
                pdb_file.write(
                    pdb_string_formats["ter"].format(
                        index=index,
                        residue=row.residue,
                        chain_id=row.chain_id,
                        residue_number=int(row.residue_number),
                    )
                )
                index += 1
        pdb_file.write(
            pdb_string_formats["ter"].format(
                index=index,
                residue=row.residue,
                chain_id=row.chain_id,
                residue_number=int(row.residue_number),
            )
        )
        if add_END:
            pdb_file.write(pdb_string_formats["end"])
