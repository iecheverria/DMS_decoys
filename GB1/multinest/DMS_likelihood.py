import numpy as np
import pandas as pd
from scipy.special import ive


class DMSLikelihood:
    """
    von Mises-Fisher likelihood for DMS data.

    log P(DMS | structures, weights) = kappa * mu_ensemble^T * dms_unit + log C_d(kappa)

    where:
      - dms_unit     : unit vector of z-scored per-position DMS signal (fixed)
      - mu_ensemble  : weighted sum of per-structure z-scored vectors, normalised
      - kappa        : concentration parameter (fixed or inferred)
    """

    def __init__(self, dms_matrix, kappa=50.0, tail_quantile=0.20):
        """
        Parameters
        ----------
        dms_matrix     : pd.DataFrame (n_aa x n_positions)
                         Rows = amino acids, columns = positions (integers)
                         Values = ln(W) fitness scores, NaN at WT positions
        kappa          : float, vMF concentration — controls dynamic range linearly
        tail_quantile  : float or None
                         Restrict likelihood to bottom X% DMS positions (most deleterious)
                         Set to None to use all positions
        """
        self.kappa         = kappa
        self.tail_quantile = tail_quantile

        # Per-position mean across all amino acids (axis=0)
        pos_means = dms_matrix.mean(axis=0)

        # Optionally restrict to high-signal (most deleterious) positions
        if tail_quantile is not None:
            threshold   = pos_means.quantile(tail_quantile)
            high_signal = pos_means[pos_means <= threshold].index
            pos_means   = pos_means[high_signal]
            print(f"Using {len(high_signal)}/{dms_matrix.shape[1]} "
                  f"high-signal positions (bottom {tail_quantile*100:.0f}%): "
                  f"{list(high_signal)}")

        # Canonical position order — all structure vectors must match this
        self.positions = list(pos_means.index)
        self.n_dim     = len(self.positions)

        # Z-score then normalise to unit vector
        dms_z         = self._zscore(pos_means)
        self.dms_unit = dms_z / np.linalg.norm(dms_z)
        self._log_norm = self._vmf_log_normaliser(self.n_dim, self.kappa)

        print(f"DMSLikelihood: {self.n_dim} positions, kappa={kappa}")
        print(f"DMS unit vector norm: {np.linalg.norm(self.dms_unit):.6f}  (should be 1.0)")

    # ── PUBLIC INTERFACE ───────────────────────────────────────────────────────

    def get_likelihood(self, structures, weights):
        """
        Compute vMF log-likelihood of DMS data given a structural ensemble.

        Parameters
        ----------
        structures : list of pd.DataFrame
            Each (n_aa x n_positions). Positions must overlap with dms_matrix.
        weights : array-like
            Ensemble weights, length = len(structures). Will be normalised.

        Returns
        -------
        float : log-likelihood
        """
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        ensemble_vec = self._ensemble_vector(structures, weights)
        norm = np.linalg.norm(ensemble_vec)
        if norm < 1e-10:
            return -np.inf

        mu = ensemble_vec / norm   # unit vector in ensemble direction

        return float(self.kappa * np.dot(mu, self.dms_unit) + self._log_norm)

    def update_kappa(self, kappa):
        """Update kappa and recompute normaliser — call from MultiNest prior transform."""
        self.kappa     = kappa
        self._log_norm = self._vmf_log_normaliser(self.n_dim, kappa)

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _structure_vector(self, matrix):
        """
        Per-position mean of a structure matrix, z-scored within that structure.
        Aligned to self.positions (canonical DMS order) before computation.
        NOT normalised to unit length — magnitude contributes to ensemble vector.
        """
        common = [p for p in self.positions if p in matrix.columns]
        if len(common) < self.n_dim:
            raise ValueError(
                f"Structure missing {self.n_dim - len(common)} positions "
                f"present in DMS data."
            )
        pos_means = matrix[common].mean(axis=0)   # axis=0: mean across amino acids
        z         = self._zscore(pos_means)
        # Return in canonical position order as numpy array
        return np.array([z[p] for p in self.positions])

    def _ensemble_vector(self, structures, weights):
        """Weighted sum of per-structure z-scored vectors."""
        vecs = np.array([self._structure_vector(s) for s in structures])
        return weights @ vecs   # shape (n_positions,)

    @staticmethod
    def _zscore(vec):
        """Z-score a pandas Series or numpy array."""
        mu  = vec.mean()
        std = vec.std()
        if std < 1e-10:
            return np.zeros(len(vec))
        return (vec - mu) / std

    @staticmethod
    def _vmf_log_normaliser(d, kappa):
        """
        Log normalising constant for vMF in d dimensions.
        Uses ive (exponentially scaled Bessel) for numerical stability.

        log C_d(kappa) = (d/2-1)*log(kappa) - (d/2)*log(2pi) - log I_{d/2-1}(kappa)
        """
        v      = d / 2.0 - 1.0
        log_Iv = np.log(ive(v, kappa) + 1e-300) + kappa
        return v * np.log(kappa) - (d / 2.0) * np.log(2.0 * np.pi) - log_Iv


# ── PRIOR TRANSFORM (standalone, for use in MultiNest) ────────────────────────

def prior_transform_weights(cube, n_structures):
    """
    Transform unit hypercube to simplex using stick-breaking construction.
    Produces a uniform distribution over the simplex.

    Parameters
    ----------
    cube : array-like of length n_structures - 1
        Sampled from [0, 1]^(K-1) by MultiNest
    n_structures : int
        Number of structures K

    Returns
    -------
    weights : np.ndarray of length n_structures, sums to 1
    """
    cube      = np.asarray(cube, dtype=float)
    weights   = np.zeros(n_structures)
    remaining = 1.0
    for i in range(n_structures - 1):
        weights[i] = cube[i] * remaining
        remaining  -= weights[i]
    weights[-1] = remaining
    return weights


# ── SANITY CHECK ──────────────────────────────────────────────────────────────

def sanity_check(lik, structures, names):
    """
    Print per-structure cosine similarity and log-likelihood.
    True structure should have highest log-lik AND highest cosine.
    """
    print(f"\n{'Name':>12}  {'||v||':>7}  {'cosine':>8}  {'log_lik':>10}  {'-log_lik':>10}")
    print("-" * 58)
    for name, struct in zip(names, structures):
        vec      = lik._structure_vector(struct)
        vec_unit = vec / np.linalg.norm(vec)
        cosine   = np.dot(vec_unit, lik.dms_unit)
        log_lik  = lik.get_likelihood([struct], [1.0])
        print(f"{name:>12}  {np.linalg.norm(vec):>7.3f}  "
              f"{cosine:>8.3f}  {log_lik:>10.3f}  {-log_lik:>10.3f}")
