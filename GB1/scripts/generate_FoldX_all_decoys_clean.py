#!/usr/bin/env python
"""
generate_FoldX_all_decoys_clean.py
==================================

Single, self-contained FoldX driver for the whole structure set: the X-ray
("true") structure plus every ColabFold decoy. Replaces the three sequential
scripts that were used originally (FoldX/FoldX_DMS.py for the X-ray,
generate_FoldX_scores.py for decoys 1-3, generate_FoldX_all_decoys.py for
decoys 6-7).

For each structure it runs the full FoldX saturation scan:
    1. BuildModel    — one point mutant per (position, amino acid)
    2. AnalyseComplex — interaction energy of every mutant complex
    3. AnalyseComplex — interaction energy of the WT complex (reference)

Output lands in per-structure directories that `generate_energy_matrices.py`
then reads:
    FoldX/                 ← X-ray ("true") structure
    FoldX_decoy_1 … _N/    ← one per file in all_decoys/ (sorted)

Each directory ends up with WT/Summary_*.fxout and pos_<pos>/Summary_*.fxout.
The Summary filename format (Summary_<prefix>_<pos>_<idx>_<AA>_<chains>.fxout)
is what the downstream parser keys on, so the output prefix is unimportant.

Chain conventions (differ between the two sources, hence per-target config):
    X-ray  1fcc_AC.pdb  — mutate chain C, analyse chains A,C
    decoys (ColabFold)  — mutate chain B, analyse chains A,B

Paths are all relative to the project root (the parent of this script's
directory); the only machine-specific setting is FOLDX_EXE below.

Usage:
    python scripts/generate_FoldX_all_decoys_clean.py            # X-ray + all decoys
    python scripts/generate_FoldX_all_decoys_clean.py --only xray
    python scripts/generate_FoldX_all_decoys_clean.py --only decoys
"""

import argparse
import glob
import os

from Bio import SeqIO
from Bio.PDB import MMCIFParser, PDBIO

# ── Configuration ────────────────────────────────────────────────────────────

# Machine-specific FoldX executable — the one setting that is not relative.
FOLDX_EXE = '/Users/ignaciaecheverria/SOFTW/foldxMac/foldx_20261231'

aa_list = list('ACDEFGHIKLMNPQRSTVWY')

# Project root = parent of this script's directory; everything else is relative.
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Mutated-chain sequence (GB1 binder). Same sequence in every structure; only
# the chain letter in the FoldX mutation string differs between targets.
FASTA_REL = 'data/1fcc_C.fasta'

# Standard working PDB name inside each target directory, and output prefix.
MODEL_PDB    = 'model.pdb'
OUTPUT_NAME  = 'pos'

# Directory holding the raw decoy structures (.pdb / .cif).
DECOY_INPUT_DIR_REL = 'all_decoys'

# X-ray ("true") structure target.
XRAY_INPUT_REL    = 'FoldX/1fcc_AC.pdb'
XRAY_WORK_DIR_REL = 'FoldX'
XRAY_MUT_CHAIN    = 'C'
XRAY_OTHER_CHAIN  = 'A'

# Decoy chain convention.
DECOY_MUT_CHAIN   = 'B'
DECOY_OTHER_CHAIN = 'A'


# ── FoldX steps ──────────────────────────────────────────────────────────────

def cif_to_pdb(cif_path, pdb_path):
    """Convert a .cif structure to .pdb using BioPython."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('model', cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)


def stage_input(input_path, work_dir):
    """Copy/convert the raw structure into work_dir/MODEL_PDB. Returns the path."""
    os.makedirs(work_dir, exist_ok=True)
    model_path = os.path.join(work_dir, MODEL_PDB)
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.cif':
        print(f'  converting {os.path.basename(input_path)} -> {MODEL_PDB}')
        cif_to_pdb(input_path, model_path)
    else:
        with open(input_path, 'rb') as src, open(model_path, 'wb') as dst:
            dst.write(src.read())
    return model_path


def write_mutant_lists(sequence, work_dir, mut_chain):
    """One individual_list_<pos>.txt per position (all 19 non-WT substitutions)."""
    for i, aa in enumerate(sequence):
        with open(os.path.join(work_dir, f'individual_list_{i+1}.txt'), 'w') as out:
            for s in (r for r in aa_list if r != aa):
                out.write(f'{aa}{mut_chain}{i+1}{s};\n')


def build_mutants(sequence, work_dir):
    """FoldX BuildModel for every position; sort outputs into pos_<pos>/ dirs."""
    model_stem = MODEL_PDB.split('.pdb')[0]
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        for i, aa in enumerate(sequence):
            sub = [r for r in aa_list if r != aa]
            print(f'  BuildModel position {i+1}/{len(sequence)}')
            os.system(f'{FOLDX_EXE} --command=BuildModel --pdb={MODEL_PDB} '
                      f'--mutant-file=individual_list_{i+1}.txt')

            pos_dir = f'pos_{i+1}'
            os.makedirs(pos_dir, exist_ok=True)
            for j, s in enumerate(sub):
                os.system(f'mv {model_stem}_{j+1}.pdb '
                          f'{pos_dir}/{OUTPUT_NAME}_{i+1}_{j+1}_{s}.pdb')
                os.system(f'mv WT_{model_stem}_{j+1}.pdb {pos_dir}/ 2>/dev/null')
            os.system(f'mv Dif_{model_stem}.fxout {pos_dir}/ 2>/dev/null')
            os.system(f'mv Raw_{model_stem}.fxout {pos_dir}/ 2>/dev/null')
            os.system(f'mv Average_{model_stem}.fxout {pos_dir}/ 2>/dev/null')
    finally:
        os.chdir(cwd)


def analyse_mutants(sequence, work_dir, mut_chain, other_chain):
    """FoldX AnalyseComplex on every mutant model."""
    cwd = os.getcwd()
    try:
        for i, aa in enumerate(sequence):
            sub = [r for r in aa_list if r != aa]
            os.chdir(os.path.join(work_dir, f'pos_{i+1}'))
            print(f'  AnalyseComplex position {i+1}/{len(sequence)}')
            for j, s in enumerate(sub):
                pdb_name = f'{OUTPUT_NAME}_{i+1}_{j+1}_{s}.pdb'
                os.system(f'{FOLDX_EXE} --command=AnalyseComplex --pdb={pdb_name} '
                          f'--analyseComplexChains={mut_chain},{other_chain} '
                          f'--complexWithDNA=false')
            os.chdir(cwd)
    finally:
        os.chdir(cwd)


def analyse_wt(work_dir, mut_chain, other_chain):
    """FoldX AnalyseComplex on the WT (unmutated) complex -> work_dir/WT/."""
    wt_dir = os.path.join(work_dir, 'WT')
    os.makedirs(wt_dir, exist_ok=True)
    with open(os.path.join(work_dir, MODEL_PDB), 'rb') as src, \
         open(os.path.join(wt_dir, MODEL_PDB), 'wb') as dst:
        dst.write(src.read())

    cwd = os.getcwd()
    os.chdir(wt_dir)
    try:
        print('  AnalyseComplex WT')
        os.system(f'{FOLDX_EXE} --command=AnalyseComplex --pdb={MODEL_PDB} '
                  f'--analyseComplexChains={mut_chain},{other_chain} '
                  f'--complexWithDNA=false')
    finally:
        os.chdir(cwd)


def process_structure(name, input_path, work_dir, sequence, mut_chain, other_chain):
    """Full FoldX scan for one structure."""
    print(f'\n=== {name}: {os.path.relpath(input_path, ROOT)} '
          f'(mutate {mut_chain}, analyse {mut_chain},{other_chain}) ===')
    stage_input(input_path, work_dir)
    write_mutant_lists(sequence, work_dir, mut_chain)
    build_mutants(sequence, work_dir)
    analyse_mutants(sequence, work_dir, mut_chain, other_chain)
    analyse_wt(work_dir, mut_chain, other_chain)


# ── Main ─────────────────────────────────────────────────────────────────────

def discover_decoys():
    """All decoy structures in all_decoys/, sorted -> [(decoy_num, abs_path), ...]."""
    decoy_dir = os.path.join(ROOT, DECOY_INPUT_DIR_REL)
    files = sorted(glob.glob(os.path.join(decoy_dir, '*.pdb')) +
                   glob.glob(os.path.join(decoy_dir, '*.cif')))
    return list(enumerate(files, start=1))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', choices=['xray', 'decoys'], default=None,
                        help='Process only the X-ray structure or only the decoys')
    args = parser.parse_args()

    # Mutated-chain sequence (shared by every structure).
    fasta = os.path.join(ROOT, FASTA_REL)
    sequence = next(SeqIO.parse(open(fasta), 'fasta')).seq
    print(f'Sequence ({len(sequence)} residues): {sequence}')

    # X-ray / true structure.
    if args.only in (None, 'xray'):
        process_structure(
            name='true (X-ray)',
            input_path=os.path.join(ROOT, XRAY_INPUT_REL),
            work_dir=os.path.join(ROOT, XRAY_WORK_DIR_REL),
            sequence=sequence,
            mut_chain=XRAY_MUT_CHAIN,
            other_chain=XRAY_OTHER_CHAIN,
        )

    # Decoys.
    if args.only in (None, 'decoys'):
        decoys = discover_decoys()
        # Mapping is decided by the sorted filenames in all_decoys/. Print it so
        # you can confirm decoy_N matches the structure you expect (in particular
        # that decoy_6 / decoy_7 stay the correct-binding-mode structures).
        print(f'\nFound {len(decoys)} decoy structures in {DECOY_INPUT_DIR_REL}/'
              f' (verify this mapping):')
        for decoy_num, input_path in decoys:
            print(f'  decoy_{decoy_num} <- {os.path.basename(input_path)}')
        for decoy_num, input_path in decoys:
            process_structure(
                name=f'decoy_{decoy_num}',
                input_path=input_path,
                work_dir=os.path.join(ROOT, f'FoldX_decoy_{decoy_num}'),
                sequence=sequence,
                mut_chain=DECOY_MUT_CHAIN,
                other_chain=DECOY_OTHER_CHAIN,
            )

    print('\nDone. Next: python scripts/generate_energy_matrices.py')


if __name__ == '__main__':
    main()
