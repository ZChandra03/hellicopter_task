import numpy as np

def _check_sigma(s):
    if not (s > 0):
        raise ValueError("sigma must be > 0")

def _normal_logpdf(x, mu, sigma):
    """Log N(x | mu, sigma^2)"""
    _check_sigma(sigma)
    z = (x - mu) / sigma
    return -0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*z*z

def _sigmoid_stable(z):
    """Numerically stable logistic."""
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def bayes_ideal_observer(x, mu1, sigma1, mu2, sigma2, prior1=0.5):
    """
    Bayesian ideal observer for two known Gaussians.
    Returns a dict with likelihoods, LLR, posteriors, and MAP decision.
    """
    if not (0.0 < prior1 < 1.0):
        raise ValueError("prior1 must be in (0,1)")
    x = np.asarray(x)

    ll1 = _normal_logpdf(x, mu1, sigma1)
    ll2 = _normal_logpdf(x, mu2, sigma2)
    llr = ll1 - ll2
    logit_prior = np.log(prior1) - np.log(1.0 - prior1)

    post1 = _sigmoid_stable(llr + logit_prior)
    post2 = 1.0 - post1

    like1 = np.exp(ll1)
    like2 = np.exp(ll2)

    decision = np.where(post1 >= 0.5, 1, 2)

    return {
        "likelihood1": like1,
        "likelihood2": like2,
        "log_likelihood_ratio": llr,
        "posterior1": post1,
        "posterior2": post2,
        "decision_map": decision
    }

if __name__ == "__main__":
    #rng = np.random.default_rng(0)
    rng = np.random
    mu1, sigma1 = -1.0, 1.0
    mu2, sigma2 = 1.0, 1.0
    prior1 = 0.5

    x_test = rng.normal(mu1, sigma1)

    mu1_det, sigma1_det = -1.0, 1.0
    mu2_det, sigma2_det = 1.0, 1.0
    prior1_det = 0.1

    out = bayes_ideal_observer(x_test, mu1_det, sigma1_det, mu2_det, sigma2_det, prior1_det)
    print("x:", x_test)
    print("P(H1|x):", np.round(out["posterior1"], 4))
    print("P(H2|x):", np.round(out["posterior2"], 4))
    print("MAP decision (1 vs 2):", out["decision_map"])
