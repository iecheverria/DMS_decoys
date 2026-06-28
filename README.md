# GB1 — Bayesian structure inference from DMS data

Infers which candidate structure(s) of the GB1 complex are consistent with deep
mutational scanning (DMS) data, by scoring FoldX ΔΔG mutational profiles against
the DMS fitness landscape and running Bayesian model selection (MultiNest).

## Pipeline

Run these three steps in order:

```bash
# 1. FoldX energies for the X-ray structure + all decoys
#    (needs FoldX installed; set FOLDX_EXE at the top of the script)
python scripts/generate_FoldX_all_decoys_clean.py

# 2. Build the (20 amino acid x 54 position) matrices
python scripts/generate_energy_matrices.py        # -> energy_matrices/*.csv

# 3. Bayesian model selection
cd multinest && python run_multinest.py
```

Steps 2–3 only need the committed `energy_matrices/*.csv`, so they run without
FoldX installed if you just want to reproduce the inference.

## Inputs

- `data/single_mutants.csv` — raw Olson et al. (2014) DMS counts.
- `data/1fcc_C.fasta` — GB1 binder sequence (the mutated chain).
- `all_decoys/` — ColabFold decoy structures (`.pdb`/`.cif`).
- `FoldX/1fcc_AC.pdb` — X-ray ("true") complex.

## Layout
```
data/                  raw DMS counts + sequence
all_decoys/            decoy structures (FoldX input)
FoldX/, FoldX_decoy_*/ FoldX output per structure
energy_matrices/       generated CSV matrices (pipeline I/O)
scripts/               generate_FoldX_all_decoys_clean.py, generate_energy_matrices.py
multinest/             run_multinest.py + likelihood/sampler modules
```

