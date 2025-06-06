import os

from Bio import Align
from CifFile import ReadCif
from openmm.app import PDBFile
from openmm.unit import angstroms, Quantity
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import Sequence

import pandas


class Molecule(object):
    def __init__(
        self,
        type,
        reference_sequence,
        original_chain_id,
        accession_code,
        parent,
        adjust_sequence=False,
    ) -> None:
        self.type = type
        self.original_chain_id = original_chain_id
        self._chain_id = original_chain_id
        self.reference_sequence = reference_sequence
        self.reference_sequence_as_list = []
        self.adjust_sequence = adjust_sequence
        self.accession_code = accession_code
        self.parent = parent

        self._residue_numbers_fixed = False
        self.residue_numbers_new2old = {}

        self.pdb_entries = {
            "record": [],
            "atom": [],
            "residue": [],
            "chain_id": [],
            "residue_number": [],
            "x": [],
            "y": [],
            "z": [],
            "occupancy": [],
            "b_factor": [],
            "element": [],
        }

        self.pdb_string_formats = {
            "coordinates": "{record:<6}{index:>5} {atom:<4} {residue:<4}{chain_id:1}{residue_number:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}          {element:>2}  \n",
            "ter": "TER   {index:>5}      {residue:<4}{chain_id:1}{residue_number:>4}\n",
            "end": "END\n",
        }

    def conclude_data_entry(self):
        self.data = pandas.DataFrame(self.pdb_entries)
        self.num_entries = len(self.data)

    @staticmethod
    def _convert_three_to_one_letter_code(sequence):
        # fmt: off
        three_to_one = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", 
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", 
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", 
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
            
            # Non-standard or ambiguous residues
            "ASX": "B",  # Aspartic acid or Asparagine
            "GLX": "Z",  # Glutamic acid or Glutamine
            "XLE": "J",  # Leucine or Isoleucine
            "SEC": "U",  # Selenocysteine
            "PYL": "O",  # Pyrrolysine
            "XAA": "X"   # Unknown or unspecified amino acid
        }

        return [three_to_one.get(AA, "X") for AA in sequence]

    def fix_residue_numbers(self):
        print(
            f"Changing residue numbers in Molecule {self.original_chain_id}:"
        )
        residues_in_structure = []
        res_number_in_structure = []

        previous_residue_number = 0
        for _, row in self.data.iterrows():
            if previous_residue_number != row.residue_number:
                residues_in_structure.append(row.residue)
                res_number_in_structure.append(
                    int(row.residue_number)
                )
            previous_residue_number = row.residue_number

        if self.type == "protein":
            residues_in_structure = (
                self._convert_three_to_one_letter_code(
                    residues_in_structure
                )
            )

        aligner = Align.PairwiseAligner(
            mode="global",
            match_score=2,
            mismatch_score=-8,
            target_internal_open_gap_score=-8,
            target_internal_extend_gap_score=-4,
            target_left_open_gap_score=-2,
            target_left_extend_gap_score=-1,
            target_right_open_gap_score=-2,
            target_right_extend_gap_score=-1,
            query_internal_open_gap_score=-2,
            query_internal_extend_gap_score=-1,
            query_left_open_gap_score=-2,
            query_left_extend_gap_score=-1,
            query_right_open_gap_score=-2,
            query_right_extend_gap_score=-1,
        )
        self.alignments = aligner.align(
            self.reference_sequence, "".join(residues_in_structure)
        )
        print(self.alignments[0])

        new_residue_numbers = (
            self.alignments[0].inverse_indices[1] + 1
        )

        # check for errors in the alignment
        for idx in range(1, len(res_number_in_structure)):
            if (
                res_number_in_structure[idx]
                - res_number_in_structure[idx - 1]
                == 1
            ) and (
                new_residue_numbers[idx]
                - new_residue_numbers[idx - 1]
                != 1
            ):
                new_residue_numbers[idx] = (
                    new_residue_numbers[idx - 1] + 1
                )

        old_data = self.data.copy()

        for res_number, new_residue_number in zip(
            res_number_in_structure, new_residue_numbers
        ):
            print(
                f"\tresidue {res_number} to {new_residue_number}",
                flush=True,
            )

            self.data.loc[
                old_data["residue_number"] == str(res_number),
                "residue_number",
            ] = str(new_residue_number)

            self.residue_numbers_new2old[new_residue_number] = (
                res_number
            )

        self.pdb_entries = self.data.to_dict(orient="list")
        self._residue_numbers_fixed = True

        return self.residue_numbers_new2old

    # def fix_residue_numbers(self):

    #     print(
    #         f"Changing residue numbers in Molecule {self.original_chain_id}:"
    #     )
    #     residues_in_structure = []
    #     res_number_in_structure = []

    #     previous_residue_number = 0
    #     for _, row in self.data.iterrows():
    #         if previous_residue_number != row.residue_number:
    #             residues_in_structure.append(row.residue)
    #             res_number_in_structure.append(
    #                 int(row.residue_number)
    #             )
    #         previous_residue_number = row.residue_number

    #     if self.type == "protein":
    #         residues_in_structure = (
    #             self._convert_three_to_one_letter_code(
    #                 residues_in_structure
    #             )
    #         )

    #     # break the chunks that appear in the structure
    #     chunks_in_structure = {
    #         "res_number": [],
    #         "residues": [],
    #     }

    #     previous_residue_number = 0
    #     for res_number, residue in zip(
    #         res_number_in_structure, residues_in_structure
    #     ):
    #         if previous_residue_number != res_number - 1:
    #             if previous_residue_number:
    #                 chunks_in_structure["res_number"].append(
    #                     chunk_res_number
    #                 )
    #                 chunks_in_structure["residues"].append(
    #                     chunk_residues
    #                 )

    #             chunk_res_number = []
    #             chunk_residues = []

    #         chunk_res_number.append(res_number)
    #         chunk_residues.append(residue)

    #         previous_residue_number = res_number
    #     chunks_in_structure["res_number"].append(chunk_res_number)
    #     chunks_in_structure["residues"].append(chunk_residues)

    #     for chunk_residues, chunk_res_number in zip(
    #         chunks_in_structure["residues"],
    #         chunks_in_structure["res_number"],
    #     ):
    #         chunk_residues = "".join(chunk_residues)

    #         new_residue_number = (
    #             self.reference_sequence.index(chunk_residues) + 1
    #         )
    #         for res_number in chunk_res_number:
    #             print(
    #                 f"\tresidue {res_number} to {new_residue_number}",
    #                 flush=True,
    #             )
    #             self.data.query(
    #                 f"residue_number == {str(res_number)}"
    #             )[
    #                 "residue_number",
    #             ] = str(
    #                 new_residue_number
    #             )

    #             self.residue_numbers_new2old[new_residue_number] = (
    #                 res_number
    #             )

    #             new_residue_number += 1

    #     self.pdb_entries = self.data.to_dict(orient="list")

    #     self._residue_numbers_fixed = True

    #     return self.residue_numbers_new2old

    @property
    def chain_id(self):
        return self._chain_id

    @chain_id.setter
    def chain_id(self, id):

        try:
            id = str(id)
        except:
            raise TypeError(
                "chain_id should be `str` or be convertible to `str`."
            )

        if len(id) == 0:
            raise ValueError("Invalid chain_id")
        elif len(id) == 1:
            pass
        elif len(id) == 2:
            raise Warning(
                "chain_id has 2 characters! Resulting `pdb` might be affected"
            )
        else:
            raise ValueError("Invalid chain_id")

        self._chain_id = id

        self.data["chain_id"] = self._chain_id
        self.pdb_entries = self.data.to_dict(orient="list")

    def save_pdb(
        self,
        filename=None,
        folder_path=None,
        ignore_HETATM=False,
        file_mode="w",
        add_END=True,
    ):
        if folder_path is None:
            folder_path = os.getcwd()
        if filename is None:
            if self._chain_id.isalpha():
                filename = (
                    f"{self.parent}.upper.{self._chain_id}.pdb"
                    if self._chain_id.isupper()
                    else f"{self.parent}.lower.{self._chain_id}.pdb"
                )
            else:
                filename = f"{self.parent}.{self._chain_id}.pdb"

        with open(
            os.path.join(folder_path, filename), file_mode
        ) as pdb_file:
            index = 1
            for _, row in self.data.iterrows():
                if row.record == "ATOM":
                    pdb_file.write(
                        self.pdb_string_formats["coordinates"].format(
                            record=row.record,
                            index=index,
                            atom=row.atom,
                            residue=row.residue,
                            chain_id=(
                                row.chain_id
                                if len(row.chain_id) == 1
                                else "."
                            ),
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
                elif row.record == "HETATM" and not ignore_HETATM:
                    pdb_file.write(
                        self.pdb_string_formats["coordinates"].format(
                            record=row.record,
                            index=index,
                            atom=row.atom,
                            residue=row.residue,
                            chain_id=(
                                row.chain_id
                                if len(row.chain_id) == 1
                                else "."
                            ),
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
            pdb_file.write(
                self.pdb_string_formats["ter"].format(
                    index=index,
                    residue=row.residue,
                    chain_id=(
                        row.chain_id
                        if len(row.chain_id) == 1
                        else "."
                    ),
                    residue_number=int(row.residue_number),
                )
            )
            if add_END:
                pdb_file.write(self.pdb_string_formats["end"])

    def complete_side_chains(
        self,
        keep_original_pdb=True,
        folder_path=None,
        filename=None,
        save_fixed_pdb=True,
    ):
        if folder_path is None:
            folder_path = os.path.join(
                os.getcwd(), f"{self.parent}.chains"
            )
        os.makedirs(folder_path, exist_ok=True)

        original_folder = os.path.join(folder_path, "original")
        os.makedirs(original_folder, exist_ok=True)

        if filename is None:
            if self._chain_id.isalpha():
                filename = (
                    f"{self.parent}.upper.{self._chain_id}.pdb"
                    if self._chain_id.isupper()
                    else f"{self.parent}.lower.{self._chain_id}.pdb"
                )
            else:
                filename = f"{self.parent}.{self._chain_id}.pdb"

        self.save_pdb(
            filename=filename,
            folder_path=original_folder,
            ignore_HETATM=False,
        )

        fixer = PDBFixer(
            filename=os.path.join(original_folder, filename)
        )

        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # fixer.sequences.append(
        #     Sequence(self._chain_id, self.reference_sequence_as_list)
        # )
        # fixer.findMissingResidues()

        # # only complete residues in the middle
        # chains = list(fixer.topology.chains())
        # keys = list(fixer.missingResidues.keys())
        # for key in keys:
        #     chain = chains[key[0]]
        #     if key[1] == 0 or key[1] == len(list(chain.residues())):
        #         del fixer.missingResidues[key]

        # try:
        # fixer.addMissingAtoms()
        # except:
        #     print(
        #         f"Couldn't complete missing residues for chain {self.chain_id}, will only complete side chains!"
        #     )
        #     fixer.missingResidues = {}
        #     fixer.addMissingAtoms()

        if save_fixed_pdb:
            fixed_folder = os.path.join(folder_path, "fixed")
            os.makedirs(fixed_folder, exist_ok=True)

            PDBFile.writeFile(
                fixer.topology,
                fixer.positions,
                open(os.path.join(fixed_folder, filename), "w"),
                keepIds=True,
            )

        self._update_molecule_from_openmm_format(
            fixer.topology, fixer.positions
        )

        if not keep_original_pdb:
            os.remove(os.path.join(original_folder, filename))

    def _update_molecule_from_openmm_format(
        self, topology, positions
    ):
        if isinstance(positions, Quantity):
            positions = positions.value_in_unit(angstroms)

        non_heterogens = PDBFile._standardResidues[:-1]
        position_index = 0

        updated_pdb_entries = {
            "record": [],
            "atom": [],
            "residue": [],
            "chain_id": [],
            "residue_number": [],
            "x": [],
            "y": [],
            "z": [],
            "occupancy": [],
            "b_factor": [],
            "element": [],
        }
        updated_occupancy = []
        updated_b_factor = []

        for chain_index, chain in enumerate(topology.chains()):
            residues = list(chain.residues())

            for residue_index, residue in enumerate(residues):
                if len(residue.name) > 3:
                    residue_name = residue.name[:3]
                else:
                    residue_name = residue.name

                if residue.name in non_heterogens:
                    record = "ATOM"
                else:
                    record = "HETATM"

                for atom in residue.atoms():
                    if atom.element is not None:
                        symbol = atom.element.symbol

                    # if (
                    #     len(atom.name) < 4
                    #     and atom.name[:1].isalpha()
                    #     and len(symbol) < 2
                    # ):
                    #     atom_name = " " + atom.name
                    # elif len(atom.name) > 4:
                    #     atom_name = atom.name[:4]
                    # else:
                    #     atom_name = atom.name

                    coords = positions[position_index]

                    updated_pdb_entries["record"].append(record)
                    updated_pdb_entries["atom"].append(atom.name)
                    updated_pdb_entries["residue"].append(
                        residue_name
                    )
                    updated_pdb_entries["chain_id"].append(
                        self.chain_id
                    )
                    updated_pdb_entries["residue_number"].append(
                        residue.id
                    )
                    updated_pdb_entries["x"].append(coords[0])
                    updated_pdb_entries["y"].append(coords[1])
                    updated_pdb_entries["z"].append(coords[2])
                    updated_pdb_entries["occupancy"].append("")
                    updated_pdb_entries["b_factor"].append("")
                    updated_pdb_entries["element"].append(symbol)

                    position_index += 1

        updated_data = pandas.DataFrame(updated_pdb_entries)

        for _, row in updated_data.iterrows():
            atom = row.atom
            residue = row.residue
            residue_number = row.residue_number

            selection = self.data.query(
                "atom == @atom & residue == @residue & residue_number == @residue_number"
            )

            if selection.empty:
                updated_occupancy.append("1.00")
                updated_b_factor.append("0.00")
            else:
                updated_occupancy.append(
                    selection.occupancy.tolist()[0]
                )
                updated_b_factor.append(
                    selection.b_factor.tolist()[0]
                )

        updated_pdb_entries["occupancy"] = updated_occupancy
        updated_pdb_entries["b_factor"] = updated_b_factor

        self.original_pdb_entries = self.pdb_entries.copy()
        self.pdb_entries = updated_pdb_entries

        self.conclude_data_entry()


class Complex(object):
    def __init__(self, name=None) -> None:
        self.molecules = {}
        self._fixed_chain_ids = {}
        self.name = name

    def load_cif(self, filename: str, fix_chain_ids=True):
        self.cif = ReadCif(filename=filename)
        cif_primary_key = self.cif.keys()[0]

        if self.name is None:
            self.name = cif_primary_key

        self.cif = self.cif[cif_primary_key]

        entity2chain = {}
        # initiate molecules
        for (
            entity_id,
            molecule_type,
            chain_ids,
            sequence,
            accession_code,
        ) in zip(
            self.cif["_entity_poly.entity_id"],
            self.cif["_entity_poly.type"],
            self.cif["_entity_poly.pdbx_strand_id"],
            self.cif["_entity_poly.pdbx_seq_one_letter_code"],
            self.cif["_struct_ref.pdbx_db_accession"],
        ):
            if entity_id not in entity2chain.keys():
                entity2chain[entity_id] = []

            if molecule_type in ["polypeptide(L)", "polypeptide(D)"]:
                fixed_molecule_type = "protein"
            elif molecule_type in [
                "polyribonucleotide",
                "polyribonucleotide",
            ]:
                fixed_molecule_type = "nucleic acid"
            else:
                fixed_molecule_type = "unknown"

            for chain_id in chain_ids.split(sep=","):
                entity2chain[entity_id].append(chain_id)

                need_to_adjust_sequence = False
                start_in_reference_sequence = self.cif[
                    "_struct_ref_seq.pdbx_auth_seq_align_beg"
                ][
                    self.cif["_struct_ref_seq.pdbx_strand_id"].index(
                        chain_id
                    )
                ]
                if (
                    int(start_in_reference_sequence) != 1
                ):  # possibly negative values in the pdb
                    need_to_adjust_sequence = True

                self.molecules[chain_id] = Molecule(
                    type=fixed_molecule_type,
                    reference_sequence="".join(
                        char for char in sequence if char.isalnum()
                    ),
                    original_chain_id=chain_id,
                    adjust_sequence=need_to_adjust_sequence,
                    parent=cif_primary_key,
                    accession_code=accession_code,
                )

        # load pdb entries
        for (
            record,
            atom,
            residue,
            chain_id,
            residue_number,
            x,
            y,
            z,
            occupancy,
            b_factor,
            element,
        ) in zip(
            self.cif["_atom_site.group_PDB"],
            self.cif["_atom_site.label_atom_id"],
            self.cif["_atom_site.label_comp_id"],
            self.cif["_atom_site.auth_asym_id"],
            self.cif["_atom_site.auth_seq_id"],
            self.cif["_atom_site.Cartn_x"],
            self.cif["_atom_site.Cartn_y"],
            self.cif["_atom_site.Cartn_z"],
            self.cif["_atom_site.occupancy"],
            self.cif["_atom_site.B_iso_or_equiv"],
            self.cif["_atom_site.type_symbol"],
        ):
            self.molecules[chain_id].pdb_entries["record"].append(
                record
            )
            self.molecules[chain_id].pdb_entries["atom"].append(atom)
            self.molecules[chain_id].pdb_entries["residue"].append(
                residue
            )
            self.molecules[chain_id].pdb_entries["chain_id"].append(
                chain_id
            )
            self.molecules[chain_id].pdb_entries[
                "residue_number"
            ].append(residue_number)
            self.molecules[chain_id].pdb_entries["x"].append(x)
            self.molecules[chain_id].pdb_entries["y"].append(y)
            self.molecules[chain_id].pdb_entries["z"].append(z)
            self.molecules[chain_id].pdb_entries["occupancy"].append(
                occupancy
            )
            self.molecules[chain_id].pdb_entries["b_factor"].append(
                b_factor
            )
            self.molecules[chain_id].pdb_entries["element"].append(
                element
            )

        # get residue name sequence
        for residue_name, entity_id in zip(
            self.cif["_entity_poly_seq.mon_id"],
            self.cif["_entity_poly_seq.entity_id"],
        ):
            for chain_id in entity2chain[entity_id]:
                self.molecules[
                    chain_id
                ].reference_sequence_as_list.append(residue_name)

        for chain_id in self.molecules.keys():

            self.molecules[chain_id].conclude_data_entry()

            if self.molecules[chain_id].adjust_sequence:
                self.molecules[chain_id].fix_residue_numbers()

        if fix_chain_ids:
            self.fix_chain_ids()

    def fix_chain_ids(self):

        # fmt: off
        chain_id_options = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", 
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "@", "!", "#", "$", "%", "&", "+", "(", ")", "-", ":", "=", "[", "]", "^", "_", "{", "}", "~"
        ]

        # fmt: on
        acession_codes = []
        original_chain_ids = [
            original_chain_id
            for original_chain_id in self.molecules.keys()
        ]

        for original_chain_id in original_chain_ids:
            acession_codes.append(
                self.molecules[original_chain_id].accession_code
                if self.molecules[
                    original_chain_id
                ].accession_code.lower()
                != self.molecules[original_chain_id].parent.lower()
                else "zzzzzz"
            )

        sorted_original_chain_ids = [
            id
            for _, id in sorted(
                zip(acession_codes, original_chain_ids)
            )
        ]

        for new_chain_id, original_chain_id in zip(
            chain_id_options, sorted_original_chain_ids
        ):
            self.molecules[original_chain_id].chain_id = new_chain_id
            self._fixed_chain_ids[original_chain_id] = new_chain_id

    def complete_side_chains(
        self,
        keep_original_pdbs=True,
        save_fixed_pdbs=True,
        folder_path=None,
        filename=None,
    ):
        print("Completing side chains...")

        num_chains = len(self.molecules.keys())
        for idx, original_chain_id in enumerate(
            self.molecules.keys()
        ):
            if self._fixed_chain_ids:
                print(
                    f"{idx+1}/{num_chains}: Processing chain {self._fixed_chain_ids[original_chain_id]}..."
                )
            else:
                print(
                    f"{idx+1}/{num_chains}: Processing chain {original_chain_id}..."
                )

            self.molecules[original_chain_id].complete_side_chains(
                keep_original_pdb=keep_original_pdbs,
                folder_path=folder_path,
                filename=filename,
                save_fixed_pdb=save_fixed_pdbs,
            )

        print("All chains processed!")

    def save_pdb(
        self, filename=None, folder_path=None, ignore_HETATM=False
    ):
        if folder_path is None:
            folder_path = os.getcwd()
        if filename is None:
            filename = f"{self.name}.pdb"

        original_chain_ids = list(self.molecules.keys())

        if self._fixed_chain_ids:
            modified_chain_ids = [
                self._fixed_chain_ids[id] for id in original_chain_ids
            ]
        else:
            modified_chain_ids = original_chain_ids.copy()

        sorted_original_chain_ids = [
            id
            for _, id in sorted(
                zip(modified_chain_ids, original_chain_ids)
            )
        ]

        num_chains = len(sorted_original_chain_ids)
        for idx, original_chain_id in enumerate(
            sorted_original_chain_ids
        ):
            if idx == 0:
                self.molecules[original_chain_id].save_pdb(
                    filename=filename,
                    folder_path=folder_path,
                    ignore_HETATM=ignore_HETATM,
                    file_mode="w",
                    add_END=False,
                )
            elif idx == num_chains - 1:
                self.molecules[original_chain_id].save_pdb(
                    filename=filename,
                    folder_path=folder_path,
                    ignore_HETATM=ignore_HETATM,
                    file_mode="a",
                    add_END=True,
                )
            else:
                self.molecules[original_chain_id].save_pdb(
                    filename=filename,
                    folder_path=folder_path,
                    ignore_HETATM=ignore_HETATM,
                    file_mode="a",
                    add_END=False,
                )
