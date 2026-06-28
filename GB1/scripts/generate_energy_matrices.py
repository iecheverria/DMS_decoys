#!/usr/bin/env python
"""
generate_energy_matrices.py
===========================

Build the (20 amino acid x N position) matrices consumed by the rest of the
pipeline, from raw inputs:

  1. DMS matrix      ← data/single_mutants.csv   (Olson et al. 2014 counts)
  2. FoldX matrices  ← FoldX/ and FoldX_decoy_*/ AnalyseComplex output

This is the clean, scripted version of the data-loading half of the development
notebook ``scripts/plot_DMS_FoldX.ipynb`` (cells reading single_mutants.csv and
parsing the FoldX Summary_*.fxout files). It writes the CSVs in
``energy_matrices/`` that ``analysis_DMS_FoldX.py`` and ``multinest/`` read.

--- Input formats ---------------------------------------------------------------

data/single_mutants.csv  (header on row 2; pass skiprows=1)
    columns: 'WT amino acid', 'Position', 'Mutation', 'Input Count', 'Selection Count'
    DMS fitness used downstream:  ln(W) = ln(Selection Count / Input Count) / 1.7287
    (1.7287 is the WT log-enrichment normaliser for this GB1 dataset.)

FoldX directories
    Each structure directory (FoldX/ for the true X-ray structure, FoldX_decoy_k/
    for decoys) contains:
        WT/Summary_*.fxout              — AnalyseComplex of the WT complex
        pos_<pos>/Summary_<...>.fxout   — AnalyseComplex of each point mutant
    The mutant Summary filename encodes position and amino acid:
        Summary_<prefix>_<pos>_<idx>_<AA>_<chains>.fxout
    Column 5 (0-indexed) of the last data line is the Interaction Energy.
    Stored value is the ΔΔG of binding:  dE = E_mutant - E_WT.

--- Output ----------------------------------------------------------------------

energy_matrices/
    dms_matrix.csv                  rows = amino acids (20), columns = positions
    matrix_true.csv                 FoldX ΔΔG, true structure
    FoldX_matrix_decoy_{1..7}.csv   FoldX ΔΔG, each decoy

All CSVs use the amino-acid letter as the index (first column = 'Mutation'),
so they must be read back with ``pd.read_csv(..., index_col=0)``.

Usage:
    python generate_energy_matrices.py                 # write to ../energy_matrices
    python generate_energy_matrices.py --output_dir /tmp/check   # dry comparison
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# Canonical amino-acid row order (charged → polar → special → hydrophobic → aromatic)
AMINO_ACIDS = ['E', 'D', 'R', 'K', 'H', 'Q', 'N', 'S', 'T', 'P',
               'G', 'C', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

# WT log-enrichment normaliser for the GB1 dataset (Olson et al. 2014)
WT_LOG_NORMALISER = 1.7287

# Mutant Summary filename: Summary_<prefix>_<pos>_<idx>_<AA>_<chains>.fxout
_MUTANT_RE = re.compile(r'Summary_.*?_(\d+)_\d+_([A-Z])_\w+\.fxout')


# ─────────────────────────────────────────────────────────────────────────────
# DMS matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_dms_matrix(single_mutants_csv):
    """ln(W) fitness matrix from the raw DMS counts. (20 x N), positions > 2."""
    D = pd.read_csv(single_mutants_csv, skiprows=1)
    D['W'] = np.log(D['Selection Count'] / D['Input Count']) / WT_LOG_NORMALISER
    D = D[D['Position'] > 2]
    matrix = (D.pivot_table(index='Mutation', columns='Position',
                            values='W', fill_value=np.nan)
              .reindex(AMINO_ACIDS))
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# FoldX matrices
# ─────────────────────────────────────────────────────────────────────────────

def _parse_interaction_energy(fxout_file):
    """Interaction Energy (column 5) from the last data line of a Summary fxout."""
    with open(fxout_file) as f:
        last_line = f.readlines()[-1]
    return float(last_line.split()[5])


def get_interaction_energies(structure_dir):
    """ΔΔG-of-binding table for every mutant in a structure directory.

    Returns a long DataFrame [Position, Mutation, dE] with dE = mutant - WT.
    Works for any Summary filename prefix (pos, 1fcc_AC, decoy, ...).
    """
    wt_files = glob.glob(os.path.join(structure_dir, 'WT', 'Summary_*.fxout'))
    if not wt_files:
        raise FileNotFoundError(f'No WT Summary file found in {structure_dir}/WT/')
    wt_energy = _parse_interaction_energy(wt_files[0])

    rows = []
    pos_dirs = sorted(
        glob.glob(os.path.join(structure_dir, 'pos_*')),
        key=lambda x: int(os.path.basename(x).split('_')[1]),
    )
    for pos_dir in pos_dirs:
        for m_file in glob.glob(os.path.join(pos_dir, 'Summary_*.fxout')):
            m = _MUTANT_RE.match(os.path.basename(m_file))
            if not m:
                continue
            pos = int(m.group(1))
            aa  = m.group(2)
            dE  = _parse_interaction_energy(m_file) - wt_energy
            rows.append([pos, aa, dE])

    E = (pd.DataFrame(rows, columns=['Position', 'Mutation', 'dE'])
         .sort_values(['Position', 'Mutation']).reset_index(drop=True))
    return E[E['Position'] > 2]


def build_foldx_matrix(structure_dir):
    """(20 x N) ΔΔG matrix for one structure directory."""
    E = get_interaction_energies(structure_dir)
    return (E.pivot_table(index='Mutation', columns='Position',
                          values='dE', fill_value=np.nan)
            .reindex(AMINO_ACIDS))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, '..'))

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=root,
                        help='Project root containing data/, FoldX/, FoldX_decoy_*/')
    parser.add_argument('--output_dir', default=os.path.join(root, 'energy_matrices'),
                        help='Where to write the *.csv matrices')
    parser.add_argument('--n_decoys', type=int, default=7, help='Number of decoys')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # DMS
    dms = build_dms_matrix(os.path.join(args.root, 'data', 'single_mutants.csv'))
    dms.to_csv(os.path.join(args.output_dir, 'dms_matrix.csv'))
    print(f'dms_matrix.csv         {dms.shape}')

    # True structure
    true_matrix = build_foldx_matrix(os.path.join(args.root, 'FoldX'))
    true_matrix.to_csv(os.path.join(args.output_dir, 'matrix_true.csv'))
    print(f'matrix_true.csv        {true_matrix.shape}')

    # Decoys
    for k in range(1, args.n_decoys + 1):
        decoy_dir = os.path.join(args.root, f'FoldX_decoy_{k}')
        if not os.path.isdir(decoy_dir):
            print(f'  (skipping decoy {k}: {decoy_dir} not found)')
            continue
        m = build_foldx_matrix(decoy_dir)
        m.to_csv(os.path.join(args.output_dir, f'FoldX_matrix_decoy_{k}.csv'))
        print(f'FoldX_matrix_decoy_{k}.csv  {m.shape}')

    print(f'\nWrote matrices to {args.output_dir}')


if __name__ == '__main__':
    main()
