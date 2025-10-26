"""
Tri-SEM: Robust Regression Framework
This file defines the main TriSEM class that integrates the Split, Extraction, and Merge stages.
"""
from split import split
from extraction import extraction
from merge import merge

class TriSEM:
    """
    Tri-SEM: A three-stage robust regression framework.
    """

    def __init__(self, p_thr=0.5, c_thr=0.5, h_thr=0.5, max_iters=20, k=2, seed=None):
        """
        Initialize the TriSEM model.

        Parameters
        ----------
        p_thr : float
            Threshold for adjacency similarity (Split stage).
        c_thr : float
            Threshold for linearity similarity (Split stage).
        h_thr : float
            Threshold for prominence criterion (Merge stage).
        max_iters : int
            Maximum number of iterations for the Split stage.
        k : int
            Initial number of clusters for the GMM in the Split stage.
        seed : int or None
            Random seed for reproducibility.
        """
        self.p_thr = p_thr
        self.c_thr = c_thr
        self.h_thr = h_thr
        self.max_iters = max_iters
        self.k = k
        self.seed = seed

        # Will be set after fitting
        self.labels_ = None                # Split stage labels
        self.partial_clean_label_ = None   # Extraction stage selected labels
        self.beta_intermediate_ = None     # Coefficients from Extraction stage
        self.beta_robust_ = None           # Final robust regression coefficients
        self.y_predict_ = None             # Predicted residuals (y - X@beta_robust)

    def fit(self, X, y):
        """
        Fit the TriSEM model to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix (including intercept).
        y : ndarray of shape (n_samples, 1)
            Response variable.

        Returns
        -------
        self : object
            Fitted model.
        """
        current_seed = self.seed

        # === Stage 1: Split ===
        labels = split(X, y,
                       k=self.k,
                       p_thr=self.p_thr,
                       c_thr=self.c_thr,
                       max_iters=self.max_iters,
                       seed=current_seed)

        # === Stage 2: Extraction ===
        partial_clean_label, beta_f = extraction(X, y, labels)

        # === Stage 3: Merge ===
        beta_robust = merge(X, y, labels,
                            partial_clean_label,
                            beta_f,
                            h_thr=self.h_thr)

        # Save results
        self.labels_ = labels
        self.partial_clean_label_ = partial_clean_label
        self.beta_intermediate_ = beta_f
        self.beta_robust_ = beta_robust
        self.y_predict_ = y - X @ beta_robust

        return self

    def predict(self, X):
        """
        Predict responses using the fitted robust coefficients.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1)
        """
        if self.beta_robust_ is None:
            raise ValueError("Model has not been fitted yet. Please call `fit(X, y)` first.")
        return X @ self.beta_robust_

    def residuals(self, X, y):
        """
        Compute residuals y - X@beta_robust.

        Returns
        -------
        residuals : ndarray
        """
        if self.beta_robust_ is None:
            raise ValueError("Model has not been fitted yet.")
        return y - X @ self.beta_robust_




# --- Example data ---
if __name__ == "__main__":
    from datasets.synthetic_data import generate_scenario1

    current_seed = 42

    X, y,clean_location = generate_scenario1(n = 5000, p = 100,contam_rate=0.3, seed=current_seed)

    model = TriSEM(p_thr=0.5, c_thr=0.5, h_thr=0.5,max_iters=20, k=2,seed=current_seed)
    model.fit(X, y)
    beta_robust = model.beta_robust_
    print("Robust coefficients:", beta_robust)
