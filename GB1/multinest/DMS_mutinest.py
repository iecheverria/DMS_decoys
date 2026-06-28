import numpy as np
import pymultinest
import json
import os
import logging

class DMSMultiNestSampler:
    """
    DMS multi-state modeling using PyMultiNest
    """
    
    def __init__(self, structures, dms_likelihood_obj, max_states=6, output_dir='./multinest_output/'):
        """
        Parameters:
        -----------
        structures : dict
            Dictionary with calculation matrices
        dms_likelihood_obj : DMS_likelihood
            DMS likelihood class with get_likelihood(structures, weights) method
        max_states : int
            Maximum number of states to consider
        output_dir : str
            Directory for MultiNest output files
        """
        self.structures = structures
        self.dms_likelihood = dms_likelihood_obj
        self.max_states = max_states
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.structure_ids = list(structures.keys())
        self.n_structures = len(self.structures_ids)
        
    def run_nested_sampling(self, n_states, n_live_points=1000):
        """
        Run nested sampling for n_states model
        """
        self.current_n_states = n_states
        
        if n_states > self.n_structures:
            return {'log_evidence': -np.inf, 'converged': False}
        
        # Parameter space: n_states weights + n_structures indicators
        n_dims = n_states + self.n_structures
        output_prefix = os.path.join(self.output_dir, f'dms_{n_states}states_')
        
        # Run MultiNest
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
        
        # Load results
        return self._load_results(output_prefix)
    
    def _log_likelihood_wrapper(self, params, n_dims, n_params):
        """
        Likelihood wrapper called by MultiNest
        """
        try:
            n_states = self.current_n_states
            
            # Extract weights and indicators
            weights = np.array(params[:n_states])
            indicators = np.array(params[n_states:n_states + self.n_structures])
            
            # Select active structure
            active_mask = indicators > 0.5
            active_structure_ids = [self.structure_ids[i] for i in range(self.n_structures) if active_mask[i]]
            
            # Check if exactly n_states are active
            if len(active_structures_ids) != n_states:
                return -1e10
            
            # Get active structures
            active_structures = [self.structures[mid] for mid in active_structures_ids]
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Call your DMS likelihood function
            log_likelihood = self.dms_likelihood.get_likelihood(active_structures, weights)
            
            return log_likelihood
            
        except Exception:
            return -1e10
    
    def _prior_transform_wrapper(self, params, n_dims, n_params):
        """
        Transform unit cube [0,1]^n to parameter space
        """
        n_states = self.current_n_states
        
        # Transform to Dirichlet weights (uniform on simplex)
        unit_weights = params[:n_states]
        gamma_samples = [-np.log(max(1e-10, u)) for u in unit_weights]
        gamma_sum = sum(gamma_samples)
        weights = [g / gamma_sum for g in gamma_samples] if gamma_sum > 0 else [1.0/n_states] * n_states
        
        # Transform to indicators (select exactly n_states)
        unit_indicators = params[n_states:n_states + self.n_structures]
        indicators = self._sample_indicators(unit_indicators, n_states)
        
        # Update params in-place
        for i, val in enumerate(weights + indicators):
            params[i] = val
    
    def _sample_indicators(self, unit_indicators, n_states):
        """
        Sample exactly n_states active structures
        """
        n_structures = len(unit_indicators)
        
        if n_states >= n_structures:
            return [1.0] * n_structures
        
        # Select n_states largest values as active
        sorted_indices = np.argsort(unit_indicators)
        indicators = [0.0] * n_structures
        for i in range(n_states):
            idx = sorted_indices[-(i+1)]
            indicators[idx] = 1.0
        
        return indicators
    
    def _load_results(self, output_prefix):
        """
        Load MultiNest results
        """
        # Load evidence
        with open(output_prefix + 'stats.dat', 'r') as f:
            lines = f.readlines()

        for line in lines:
            if "Nested Sampling Global Log-Evidence" in line:
                vals = line.split()
                log_evidence = float(vals[5])
                log_evidence_error = float(vals[7])
                break

        print(log_evidence)
        return {
            'log_evidence': log_evidence,
            'log_evidence_error': log_evidence_error,
            'converged': True
        }
    
    def compare_models(self, max_states=None, n_live_points=2000):
        """
        Compare models with different numbers of states
        """
        if max_states is None:
            max_states = min(self.max_states, self.n_structures)
        
        self.logger.info(f"Comparing models with 1 to {max_states} states...")
        
        results = {}
        log_evidences = {}
        
        for n_states in range(1, max_states + 1):
            self.logger.info(f"Running {n_states}-state model...")
            
            result = self.run_nested_sampling(n_states, n_live_points)
            results[n_states] = result
            print('Result', results)
            log_evidences[n_states] = result['log_evidence']
            
            self.logger.info(f"{n_states} states: log_evidence = {result['log_evidence']:.3f}")
        
        # Find best model
        valid_models = {k: v for k, v in log_evidences.items() if v > -1e9}
        best_n_states = max(valid_models.keys(), key=lambda k: valid_models[k])
        best_log_evidence = log_evidences[best_n_states]
        
        # Compute Bayes factors
        bayes_factors = {}
        for n_states, log_evidence in valid_models.items():
            if n_states != best_n_states:
                bayes_factors[f"{best_n_states}_vs_{n_states}"] = best_log_evidence - log_evidence
        
        self.logger.info(f"Best model: {best_n_states} states")
        
        comparison_results = {
            'model_results': results,
            'log_evidences': log_evidences,
            'best_n_states': best_n_states,
            'bayes_factors': bayes_factors
        }
        
        # Save results
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump({k: v for k, v in comparison_results.items() if k != 'model_results'}, f, indent=2)
        
        return comparison_results

    def get_best_weights(self, n_states=None):
        """
        Get the weights for the best model
        """
        if n_states is None:
            # Load comparison results to find best model
            results_file = os.path.join(self.output_dir, 'results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    comparison_results = json.load(f)
                n_states = comparison_results['best_n_states']
            else:
                raise ValueError("No comparison results found. Run compare_models() first.")
    
        # Load posterior samples for the best model
        output_prefix = os.path.join(self.output_dir, f'dms_{n_states}states_')
    
        try:
            # Load posterior samples
            post_file = output_prefix + 'post_equal_weights.dat'
            posterior_samples = np.loadtxt(post_file)
            print('ps', posterior_samples, type(posterior_samples))
            print(f"Posterior samples shape: {posterior_samples.shape}")
            print(f"First few rows:\n{posterior_samples[:3]}")
            
            # Extract parameter samples (exclude weight and likelihood columns)
            samples = posterior_samples[:,:-2]
            likelihoods = -0.5 * samples[:, -1]
    
            # Find maximum likelihood sample
            best_sample_idx = np.argmax(likelihoods)
            best_params = posterior_samples[best_sample_idx,:]
            
            # Extract weights (first n_states parameters)
            weights = best_params[:n_states]
            weights = weights / np.sum(weights)  # Check normalization
            print('hola2')
            # Extract indicators to see which structures are active
            indicators = best_params[n_states:n_states + self.n_structures]
            active_mask = indicators > 0.5
            active_structure_ids = [self.structure_ids[i] for i in range(self.n_structures) if active_mask[i]]

            print(likelihoods, samples, posterior_samples[best_sample_idx,:])
            
            return {
                'n_states': n_states,
                'weights': weights,
                'active_structures_ids': active_structure_ids,
                'max_log_likelihood': posterior_samples[best_sample_idx,-1]
            }
        
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
