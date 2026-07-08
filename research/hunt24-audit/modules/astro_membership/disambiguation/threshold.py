# threshold.py         # 策略 2: 卡方阈值截断

import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from ..base import BaseDisambiguation

class ThresholdGmmDisambiguation(BaseDisambiguation):
    def __init__(self, sigma_cutoff=3.0, **kwargs):
        self.sigma_cutoff = sigma_cutoff

    def fit_predict(self, df_all, df_seeds, features):
        X_all = df_all[features].values
        X_seeds = df_seeds[features].values
        
        gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=42)
        gmm.fit(X_seeds)
        
        n_features = len(features)
        confidence = 1.0 - (2.0 * (1.0 - chi2.cdf(self.sigma_cutoff, 1)))
        chi2_cutoff = chi2.ppf(confidence if confidence < 1 else 0.9973, df=n_features)
        
        log_det_cov = np.log(np.linalg.det(gmm.covariances_[0]))
        score_threshold = -0.5 * (chi2_cutoff + n_features * np.log(2 * np.pi) + log_det_cov)
        
        scores = gmm.score_samples(X_all)
        
        df_result = df_all.copy()
        df_result['prob'] = np.exp(scores - np.max(scores))
        df_result['is_member'] = scores >= score_threshold
        return df_result