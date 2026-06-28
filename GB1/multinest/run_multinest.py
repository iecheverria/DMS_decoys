import pandas as pd
from DMS_likelihood import DMSLikelihood, sanity_check
from DMS_multinest import DMSMultiNestSampler

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
dms_matrix = pd.read_csv('../energy_matrices/dms_matrix.csv', index_col=0)

e_matrix_true    = pd.read_csv('../energy_matrices/matrix_true.csv',    index_col=0)
e_matrix_decoy_1 = pd.read_csv('../energy_matrices/FoldX_matrix_decoy_1.csv', index_col=0)
e_matrix_decoy_2 = pd.read_csv('../energy_matrices/FoldX_matrix_decoy_2.csv', index_col=0)
e_matrix_decoy_3 = pd.read_csv('../energy_matrices/FoldX_matrix_decoy_3.csv', index_col=0)
e_matrix_decoy_6 = pd.read_csv('../energy_matrices/FoldX_matrix_decoy_6.csv', index_col=0)
e_matrix_decoy_7 = pd.read_csv('../energy_matrices/FoldX_matrix_decoy_7.csv', index_col=0)
# Note: decoy_4 and decoy_5 excluded — failed physical plausibility filter

# BUG FIX: original used .iloc[:,1:] which only works if CSV has an unnamed
# index column. Using index_col=0 on read_csv is cleaner and safer.

# ── INITIALISE LIKELIHOOD ─────────────────────────────────────────────────────
kappa          = 50.0
tail_quantile  = 0.20

dms_obj = DMSLikelihood(
    dms_matrix,
    kappa=kappa,
    tail_quantile=tail_quantile
)

# ── SANITY CHECK: single-structure likelihoods ────────────────────────────────
structure_list = [e_matrix_true,
                  e_matrix_decoy_1, e_matrix_decoy_2, e_matrix_decoy_3,
                  e_matrix_decoy_6, e_matrix_decoy_7]
names = ['true', 'decoy_1', 'decoy_2', 'decoy_3', 'decoy_6', 'decoy_7']

print("\n=== Sanity check: single-structure likelihoods ===")
sanity_check(dms_obj, structure_list, names)

# ── MULTINEST SETUP ───────────────────────────────────────────────────────────
# Accepted structures only (decoy_4 and decoy_5 rejected)
structure_dict = {
    'true'   : e_matrix_true,
    'decoy_1': e_matrix_decoy_1,
    'decoy_2': e_matrix_decoy_2,
    'decoy_3': e_matrix_decoy_3,
    'decoy_6': e_matrix_decoy_6,
    'decoy_7': e_matrix_decoy_7,
}

sampler = DMSMultiNestSampler(
    structures=structure_dict,
    dms_likelihood_obj=dms_obj,
    max_states=3,                    # start conservative
    output_dir='./multinest_output/'
)

# ── OPTION A: Full model comparison (1 to max_states) ────────────────────────
results = sampler.compare_models(max_states=3, n_live_points=400)

print("\n=== Model comparison results ===")
for n_states, evidence in results['log_evidences'].items():
    print(f"  {n_states} state(s): log_evidence = {evidence:.3f}")

print(f"\nBest model: {results['best_n_states']} state(s)")

print("\nBayes factors (ln scale, >2.3 = strong evidence for best model):")
for comparison, bf in results['bayes_factors'].items():
    print(f"  {comparison}: {bf:.3f}")

# ── OPTION B: Quick single-run test (true structure only, 1 state) ────────────
# Uncomment to test the pipeline before running full comparison
# result_1state = sampler.run_nested_sampling(
#     n_states=1,
#     structure_subset=['true'],
#     n_live_points=400
# )
# print(f"\n1-state (true only): log_evidence = {result_1state['log_evidence']:.3f}")

# ── GET BEST WEIGHTS ──────────────────────────────────────────────────────────
best = sampler.get_best_weights()
print("\nMAP weights:")
for name, w in zip(best['structure_subset'], best['map_weights']):
    print(f"  {name}: {w:.4f}")

print("\nPosterior mean weights:")
for name, w in zip(best['structure_subset'], best['mean_weights']):
    print(f"  {name}: {w:.4f}")


print("\n=== Best model weights ===")
print(best)


