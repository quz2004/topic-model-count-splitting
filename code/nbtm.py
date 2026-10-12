"""Count-based neural topic model (NTM) + count splitting for scRNA-seq.

Model (cell i, gene g, topic k):
    z_i ~ N(mu(x_i), diag(sigma^2(x_i))),  theta_i = softmax(z_i)
    p_ig = softmax_g( (theta_i @ B)_g + b_g + gamma_{batch(i), g} )
    x_i | lib_i ~ Multinomial(lib_i, p_i)
B (K x G) are free topic-gene logits, b the gene background (replaces TF-IDF), gamma optional batch intercepts.
"""
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F


# ---------------------------------------------------------------- count splitting
def split_counts(X, eps=0.5, seed=0, size=None):
    """Split integer counts X (sparse) into (train, test) with train+test == X.

    size=None : binomial thinning, exact independence under Poisson.
    size=r    : beta-binomial thinning, exact independence under NB(mean, size=r)
                (r scalar or per-gene array; data thinning, Neufeld et al. 2023).
    """
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(X)
    X.sum_duplicates()
    x = X.data.astype(np.int64)
    if size is None:
        p = eps
    else:
        r = np.broadcast_to(np.asarray(size, dtype=float), (X.shape[1],))
        rg = r[X.indices]
        p = rng.beta(eps * rg, (1 - eps) * rg)
    x1 = rng.binomial(x, p)
    tr = sp.csr_matrix((x1, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    te = sp.csr_matrix((x - x1, X.indices.copy(), X.indptr.copy()), shape=X.shape)  # copies: eliminate_zeros is in-place
    tr.eliminate_zeros(); te.eliminate_zeros()
    return tr, te


# ---------------------------------------------------------------- model
class NTM(nn.Module):
    def __init__(self, n_genes, n_topics=20, n_batches=0, hidden=256, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.mu = nn.Linear(hidden, n_topics)
        self.logvar = nn.Linear(hidden, n_topics)
        self.B = nn.Parameter(torch.randn(n_topics, n_genes) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_genes))
        self.gamma = nn.Embedding(n_batches, n_genes) if n_batches else None
        if self.gamma is not None:
            nn.init.zeros_(self.gamma.weight)

    @staticmethod
    def prep(x):  # library-size-normalized log input; comparable across full/half-depth data
        lib = x.sum(1, keepdim=True).clamp(min=1)
        return torch.log1p(x / lib * 1e3)

    def encode(self, x):
        h = self.enc(self.prep(x))
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def decode_logp(self, theta, batch=None):
        logits = theta @ self.B + self.b
        if self.gamma is not None and batch is not None:
            logits = logits + self.gamma(batch)
        return F.log_softmax(logits, dim=-1)

    def forward(self, x, batch=None, beta_kl=1.0):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        theta = F.softmax(z, -1)
        nll = -(x * self.decode_logp(theta, batch)).sum(1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        return (nll + beta_kl * kl).mean(), nll.mean(), kl.mean()


def device():
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'


def fit_ntm(X, n_topics=20, batch=None, epochs=60, bs=512, lr=2e-3, seed=0, hidden=256,
            kl_warmup=10, verbose=False, dev=None):
    """Fit on sparse count matrix X (cells x genes). batch: int array or None. Returns trained model."""
    dev = dev or device()
    torch.manual_seed(seed); np.random.seed(seed)
    X = sp.csr_matrix(X, dtype=np.float32)
    nb = int(batch.max()) + 1 if batch is not None else 0
    m = NTM(X.shape[1], n_topics, nb, hidden).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = X.shape[0]
    bt = torch.as_tensor(batch, dtype=torch.long) if batch is not None else None
    for ep in range(epochs):
        m.train()
        beta_kl = min(1.0, (ep + 1) / kl_warmup)
        perm = np.random.permutation(n); tot = 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = torch.as_tensor(X[idx].toarray()).to(dev)
            bb = bt[idx].to(dev) if bt is not None else None
            loss, nll, kl = m(xb, bb, beta_kl)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f'ep {ep} loss {tot / n:.2f}')
    return m.eval()


@torch.no_grad()
def encode_theta(model, X, bs=2048, dev=None):
    """Topic proportions (posterior mean of z, softmaxed) for sparse counts X."""
    dev = dev or next(model.parameters()).device
    X = sp.csr_matrix(X, dtype=np.float32); out = []
    for s in range(0, X.shape[0], bs):
        xb = torch.as_tensor(X[s:s + bs].toarray()).to(dev)
        out.append(F.softmax(model.encode(xb)[0], -1).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def topic_gene(model):
    """Topic-gene probabilities (K x G), background-adjusted: softmax(B_k + b)."""
    return F.softmax(model.B + model.b, -1).cpu().numpy()


# ---------------------------------------------------------------- inference helpers
def lognorm(X, scale=1e4):
    X = sp.csr_matrix(X, dtype=np.float64)
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    Y = sp.diags(scale / lib) @ X
    Y.data = np.log1p(Y.data)
    return Y


def gene_topic_corr_test(theta, Y, covars=None):
    """Pearson correlation of every topic column of theta (n x K) with every gene in Y (n x G, sparse/dense).
    Returns r (K x G) and two-sided p-values (K x G) via t-distribution (cell-level; used to demonstrate
    calibration, not as the final donor-level test)."""
    from scipy import stats
    Yd = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
    n = Yd.shape[0]
    if covars is not None:  # partial out nuisance covariates (e.g. log library size, donor) from both sides
        C = np.column_stack([np.ones(n), covars])
        Q, _ = np.linalg.qr(C)
        theta = theta - Q @ (Q.T @ theta); Yd = Yd - Q @ (Q.T @ Yd)
        n = n - (Q.shape[1] - 1)
    T = (theta - theta.mean(0)) / (theta.std(0) + 1e-12)
    Z = (Yd - Yd.mean(0)) / (Yd.std(0) + 1e-12)
    r = T.T @ Z / n
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    return r, 2 * stats.t.sf(np.abs(t), n - 2)


def gene_topic_score_test(theta, X, covars=None):
    """Quasi-Poisson score test of H0: no association between gene g counts and topic usage theta_k (given library size).
    Vectorized over topics (K) and genes (G). Returns z (K x G) and two-sided p (K x G).
    mu_ig = lib_i * rate_g under H0; dispersion phi_g estimated by Pearson chi2; theta (and covars) residualized."""
    from scipy import stats
    X = sp.csr_matrix(X, dtype=np.float64); n = X.shape[0]
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    rate = np.asarray(X.sum(0)).ravel() / lib.sum()
    C = np.ones((n, 1)) if covars is None else np.column_stack([np.ones(n), covars])
    # residualize on covariates with lib weights, then center
    w = lib / lib.mean(); sw = np.sqrt(w)[:, None]
    Q, _ = np.linalg.qr(C * sw)
    Tt = theta * sw; Tt = Tt - Q @ (Q.T @ Tt); Tt = Tt / sw                # weighted-orthogonalized theta
    Xd = X.toarray(); M = lib[:, None] * rate[None, :]
    R = Xd - M
    U = Tt.T @ R                                                          # K x G
    V = (Tt ** 2).T @ M                                                   # K x G (Poisson variance)
    phi = np.maximum(((R ** 2) / np.maximum(M, 1e-12)).sum(0) / (n - C.shape[1]), 1.0)
    z = U / np.sqrt(V * phi[None, :] + 1e-12)
    return z, 2 * stats.norm.sf(np.abs(z))
