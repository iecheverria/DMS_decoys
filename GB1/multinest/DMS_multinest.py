import numpy as np
import pymultinest
import json
import os
import logging

from DMS_likelihood import prior_transform_weights


class DMSMultiNestSampler:
    """
    DMS multi-state modelling using PyMultiNest.

    Design
    ------
    For a K-state model, MultiNest samples K-1 parameters (stick-breaking
    weights; the K-th weight is determined). The active structures for each
    run are fixed externally via run_nested_sampling(structure_subset=...).
    Structure selection (which K structures to use) is handled by exhaustive
    enumeration in compare_models(), not by binary indicators inside
    MultiNest — indicator variables inside the sampler cause discontinuous
    likelihoods and poor convergence.
    """

    def __init__(self, structures, dms_likelihood_obj,
                 max_states=4, output_dir='./multinest_output/'):
        """
        Parameters
        ----------
        structures : dict
            {name: DataFrame} — all candidate structures
        dms_likelihood_obj : DMSLikelihood
            Initialised likelihood object with get_likelihood(structures, weights)
        max_states : int
            Maximum number of ensemble states to consider
        output_dir : str
            Directory for MultiNest output files
        """
        self.structures     = structures
        self.dms_likelihood = dms_likelihood_obj
        self.max_states     = max_states
        self.output_dir     = output_dir
        self.logger         = logging.getLogger(__name__)

        os.makedirs(output_dir, exist_ok=True)

        # BUG FIX: was self.structures_ids (typo)
        self.structure_ids = list(structures.keys())
        self.n_structures  = len(self.structure_ids)

        # State set during each MultiNest run — used by wrappers
        self._active_structures = None
        self._n_active          = None

    # ── PUBLIC INTERFACE ───────────────────────────────────────────────────────

    def run_nested_sampling(self, n_states, structure_subset=None,
                            n_live_points=400):
        """
        Run nested sampling for a fixed n_states model.

        Parameters
        ----------
        n_states : int
            Number of active macrostates
        structure_subset : list of str or None
            Names of structures to use. If None, uses all structures.
        n_live_points : int
            MultiNest live points — 400 is reasonable for weight inference
        """
        if structure_subset is None:
            structure_subset = self.structure_ids

        if n_states > len(structure_subset):
            self.logger.warning(
                f"n_states={n_states} > n_structures={len(structure_subset)}, skipping.")
            return {'log_evidence': -np.inf, 'log_evidence_error': np.inf,
                    'converged': False}

        # Store active structures for use inside the wrappers
        self._active_structures = [self.structures[k] for k in structure_subset]
        self._n_active          = len(structure_subset)
        self.current_n_states   = n_states

        # K-1 free parameters (stick-breaking simplex)
        n_dims         = n_states - 1 if n_states > 1 else 1
        output_prefix  = os.path.join(
            self.output_dir,
            f'dms_{n_states}states_{"_".join(structure_subset)}_'
        )

        self.logger.info(
            f"Running {n_states}-state model with structures: {structure_subset}")

        pymultinest.run(
            LogLikelihood=self._log_likelihood_wrapper,
            Prior=self._prior_transform_wrapper,
            n_dims=n_dims,
            outputfiles_basename=output_prefix,
            verbose=False,
            resume=False,
            n_live_points=n_live_points,
            evidence_tolerance=0.5,
            sampling_efficiency=0.8,
            multimodal=True
        )

        result = self._load_results(output_prefix)
        result['structure_subset'] = structure_subset
        return result

    def compare_models(self, max_states=None, n_live_points=400):
        """
        Compare models with 1 to max_states active structures.
        For K>1, all combinations of K structures from the full set are tried
        and the best evidence per K is retained.

        Returns
        -------
        dict with keys: model_results, log_evidences, best_n_states, bayes_factors
        """
        from itertools import combinations

        if max_states is None:
            max_states = min(self.max_states, self.n_structures)

        self.logger.info(f"Comparing models with 1 to {max_states} states...")

        results      = {}
        log_evidences = {}

        for n_states in range(1, max_states + 1):
            best_result = None
            best_log_evidence = -np.inf

            # Try all combinations of n_states structures
            for subset in combinations(self.structure_ids, n_states):
                subset = list(subset)
                self.logger.info(
                    f"  {n_states} states, subset={subset}")
                result = self.run_nested_sampling(
                    n_states, structure_subset=subset,
                    n_live_points=n_live_points)

                if result['converged'] and result['log_evidence'] > best_log_evidence:
                    best_log_evidence = result['log_evidence']
                    best_result       = result

            results[n_states]      = best_result
            log_evidences[n_states] = best_log_evidence
            self.logger.info(
                f"{n_states} states: best log_evidence = {best_log_evidence:.3f} "
                f"(subset={best_result['structure_subset'] if best_result else 'none'})")

        # Best model by evidence
        valid = {k: v for k, v in log_evidences.items() if v > -1e9}
        best_n_states      = max(valid, key=valid.get)
        best_log_evidence  = log_evidences[best_n_states]

        # Bayes factors relative to best model
        bayes_factors = {
            f"{best_n_states}_vs_{k}": best_log_evidence - v
            for k, v in valid.items() if k != best_n_states
        }

        self.logger.info(f"Best model: {best_n_states} states")

        best_subsets = {}
        for n_states in range(1, max_states + 1):
            if results[n_states] and results[n_states].get('converged'):
                best_subsets[n_states] = results[n_states]['structure_subset']

        comparison_results = {
            'model_results'  : results,
            'log_evidences'  : log_evidences,
            'best_n_states'  : best_n_states,
            'bayes_factors'  : bayes_factors,
            'best_subsets'  : best_subsets, 
        }

        # Save serialisable subset
        # Save — best_subsets is serialisable (list of strings)
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(comparison_results, f, indent=2)
        #with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
        #    json.dump(
        #        {k: v for k, v in comparison_results.items()
        #         if k != 'model_results'},
        #        f, indent=2
         #   )

        return comparison_results

    def get_best_weights(self, n_states=None, structure_subset=None):
        if n_states is None:
            results_file = os.path.join(self.output_dir, 'results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    comparison_results = json.load(f)
                n_states = comparison_results['best_n_states']
                # model_results is not saved to JSON — reconstruct subset from
                # best_subsets if saved, otherwise fall back to first N structure ids
                best_subsets = comparison_results.get('best_subsets', {})
                structure_subset = best_subsets.get(
                    str(n_states), self.structure_ids[:n_states])
        else:
            raise ValueError(
                "No comparison results found. Run compare_models() first.")
    

        if structure_subset is None:
            structure_subset = self.structure_ids[:n_states]

        output_prefix = os.path.join(
            self.output_dir,
            f'dms_{n_states}states_{"_".join(structure_subset)}_'
        )

        post_file = output_prefix + 'post_equal_weights.dat'
        if not os.path.exists(post_file):
            raise FileNotFoundError(f"Posterior file not found: {post_file}")

        # Columns: [param_0, ..., param_{K-2}, log_like, log_prior_vol]
        # BUG FIX: original code sliced incorrectly and used -0.5*samples
        posterior = np.loadtxt(post_file)
        print(f"Posterior shape: {posterior.shape}")

        # Last two columns are log-likelihood and log-prior-volume
        params     = posterior[:, :-2]    # weight parameters only
        log_likes  = posterior[:, -2]     # log-likelihood column
        best_idx   = np.argmax(log_likes)
        best_params = params[best_idx]

        # Recover weights via stick-breaking
        n_dims = n_states - 1 if n_states > 1 else 1
        if n_states == 1:
            weights = np.array([1.0])
        else:
            weights = prior_transform_weights(best_params[:n_dims], n_states)

        print(f"\n=== Best weights ({n_states} states) ===")
        for name, w in zip(structure_subset, weights):
            print(f"  {name}: {w:.4f}")
        print(f"  Max log-likelihood: {log_likes[best_idx]:.3f}")

        # Posterior mean weights
        if n_states == 1:
            mean_weights = np.array([1.0])
        else:
            mean_weights = np.array([
                prior_transform_weights(params[i, :n_dims], n_states)
                for i in range(len(params))
            ]).mean(axis=0)

        print(f"\n=== Posterior mean weights ===")
        for name, w in zip(structure_subset, mean_weights):
            print(f"  {name}: {w:.4f}")

        return {
            'n_states'           : n_states,
            'structure_subset'   : structure_subset,
            'map_weights'        : weights,
            'mean_weights'       : mean_weights,
            'max_log_likelihood' : log_likes[best_idx],
        }

    # ── MULTINEST WRAPPERS ────────────────────────────────────────────────────

    def _log_likelihood_wrapper(self, params, n_dims, n_params):
        """Called by MultiNest at each sample."""
        try:
            n_states = self.current_n_states

            if n_states == 1:
                weights = np.array([1.0])
            else:
                weights = prior_transform_weights(
                    np.array(params[:n_states - 1]), n_states)

            log_lik = self.dms_likelihood.get_likelihood(
                self._active_structures[:n_states], weights)

            return float(log_lik) if np.isfinite(log_lik) else -1e300

        except Exception as e:
            self.logger.debug(f"Likelihood evaluation failed: {e}")
            return -1e300

    def _prior_transform_wrapper(self, params, n_dims, n_params):
        """
        In-place prior transform: unit cube → stick-breaking weights.
        For K=1 there are no free parameters; MultiNest still calls this.
        """
        n_states = self.current_n_states
        if n_states == 1:
            return   # nothing to transform

        # params[:K-1] are already in [0,1] — stick-breaking uses them directly
        # No transformation needed on the cube values; mapping happens in
        # prior_transform_weights at likelihood evaluation time.
        pass   # params modified in-place by MultiNest; no remapping needed here

    # ── RESULTS LOADING ───────────────────────────────────────────────────────

    def _load_results(self, output_prefix):
        """Parse MultiNest stats file for log-evidence."""
        stats_file = output_prefix + 'stats.dat'
        if not os.path.exists(stats_file):
            self.logger.error(f"Stats file not found: {stats_file}")
            return {'log_evidence': -np.inf, 'log_evidence_error': np.inf,
                    'converged': False}

        log_evidence       = -np.inf
        log_evidence_error = np.inf

        with open(stats_file, 'r') as f:
            for line in f:
                if 'Nested Sampling Global Log-Evidence' in line:
                    parts              = line.split()
                    # Format: "... Log-Evidence  <val>  +/-  <err>"
                    log_evidence       = float(parts[-3])
                    log_evidence_error = float(parts[-1])
                    break

        return {
            'log_evidence'      : log_evidence,
            'log_evidence_error': log_evidence_error,
            'converged'         : log_evidence > -1e9,
        }
