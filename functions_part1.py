import numpy as np
import scipy.optimize as opt

# Fractional Brownian Motion generation methods

def davies_harte(T, N, H, seed=None):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    g = [gamma(k,H) for k in range(0,N)];    r = g + [0] + g[::-1][0:N-1]

    # Step 1 (eigenvalues)
    j = np.arange(0,2*N);   k = 2*N-1
    lk = np.fft.fft(r*np.exp(2*np.pi*complex(0,1)*k*j*(1/(2*N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2*N,2), dtype=np.complex128); 
    Vj[0,0] = np.random.standard_normal();  Vj[N,0] = np.random.standard_normal()
    
    for i in range(1,N):
        Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*N-i][0] = Vj1;    Vj[2*N-i][1] = Vj2
    
    # Step 3 (compute Z)
    wk = np.zeros(2*N, dtype=np.complex128)   
    wk[0] = np.sqrt((lk[0]/(2*N)))*Vj[0][0];          
    wk[1:N] = np.sqrt(lk[1:N]/(4*N))*((Vj[1:N].T[0]) + (complex(0,1)*Vj[1:N].T[1]))       
    wk[N] = np.sqrt((lk[0]/(2*N)))*Vj[N][0]       
    wk[N+1:2*N] = np.sqrt(lk[N+1:2*N]/(4*N))*(np.flip(Vj[1:N].T[0]) - (complex(0,1)*np.flip(Vj[1:N].T[1])))
    
    Z = np.fft.fft(wk);     fGn = Z[0:N] 
    fBm = np.cumsum(fGn)*(N**(-H))
    fBm = (T**H)*(fBm)
    path = np.array([0] + list(fBm))
    
    if seed is not None:
        np.random.set_state(state)
    
    return path

def cholesky_fbm(T, N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    
    L = np.zeros((N,N))
    X = np.zeros(N)
    V = np.random.standard_normal(size=N)

    L[0,0] = 1.0
    X[0] = V[0]
    
    L[1,0] = gamma(1,H)
    L[1,1] = np.sqrt(1 - (L[1,0]**2))
    X[1] = np.sum(L[1,0:2] @ V[0:2])
    
    for i in range(2,N):
        L[i,0] = gamma(i,H)
        
        for j in range(1, i):         
            L[i,j] = (1/L[j,j])*(gamma(i-j,H) - (L[i,0:j] @ L[j,0:j]))

        L[i,i] = np.sqrt(1 - np.sum((L[i,0:i]**2))) 
        X[i] = L[i,0:i+1] @ V[0:i+1]

    fBm = np.cumsum(X)*(N**(-H))
    return (T**H)*(fBm)

# Normalized p-variation calculation

def normalized_p_variation(L, K, p, t, X):
    """
    Computes the normalized p-variation of a process X over time t
    
    args:
        X: array of length L+1, sampled at i*T/L
        L: fine frequency
        K: block frequency (must divide L)
        p: variation power
        t: total time
    """
    
    X = np.asarray(X)
    assert X.ndim == 1
    assert len(X) == L + 1
    assert L % K == 0
    
    m = L // K                 # fine steps per block
    dtK = t / K                # coarse time step

    dX_L = np.diff(X)          # length L, fine increments

    # reshape fine increments into K blocks, each of length m
    dX_L_blocks = dX_L.reshape(K, m)

    # block sums of |Δ^L X|^p over each coarse interval
    S = np.sum(np.abs(dX_L_blocks)**p, axis=1)   # length K

    # coarse increments Δ^K X = sum of fine increments in each block
    dX_K = np.sum(dX_L_blocks, axis=1)           # length K

    eps = 1e-15
    W = np.sum(((np.abs(dX_K)**p) / S) * dtK)
    return W

# Variation index and roughness index estimation

def estimate_p_hat(X, L, K, T=1.0, p_lo=1.01, p_hi=10.0, tol=1e-6, maxit=60):
    """
    Solve W(L,K,p,T,X) = T for p using bisection.
    Returns p_hat and H_hat=1/p_hat.
    """
    def f(p):
        return normalized_p_variation(L, K, p, T, X) - T

    flo = f(p_lo)
    fhi = f(p_hi)

    # If no sign change, you can widen p_hi or inspect monotonicity on a grid.
    if flo * fhi > 0:
        raise ValueError(f"No root bracketed: f(p_lo)={flo}, f(p_hi)={fhi}. Try widening bounds.")

    lo, hi = p_lo, p_hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            p_hat = mid
            return p_hat, 1.0 / p_hat
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    p_hat = 0.5 * (lo + hi)
    return p_hat, 1.0 / p_hat

def estimate_p_hat_robust(
    X, L, K, T=1.0,
    p_lo=1.01, p_hi=20.0,
    expand_factor=1.8, max_expand=25,
    tol=1e-6, maxit=80,
    grid_fallback=400, p_max_fallback=200.0
):
    """
    Robustly solve W(L,K,p,T,X)=T for p.

    - First tries bisection with an automatically expanded bracket.
    - If no sign change is found, falls back to minimizing |W(p)-T| on a grid,
      then refines locally with a small bisection around the best grid point if possible.

    Returns (p_hat, H_hat, info_dict).
    """
    def W(p):
        return normalized_p_variation(L, K, p, T, X)

    def f(p):
        return W(p) - T

    lo, hi = float(p_lo), float(p_hi)
    flo, fhi = f(lo), f(hi)

    # 1) Auto-expand bracket if needed
    n_expand = 0
    while flo * fhi > 0 and n_expand < max_expand:
        # expand on the side that seems "less promising"
        hi *= expand_factor
        fhi = f(hi)
        n_expand += 1

    bracketed = (flo * fhi <= 0)

    # 2) If bracketed, do bisection
    if bracketed:
        for it in range(maxit):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)

            if abs(fmid) < tol:
                p_hat = mid
                return p_hat, 1.0 / p_hat, {
                    "method": "bisection",
                    "expanded": n_expand,
                    "iters": it + 1,
                    "bracket": (lo, hi),
                    "f(lo)": flo,
                    "f(hi)": fhi
                }

            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid

        p_hat = 0.5 * (lo + hi)
        return p_hat, 1.0 / p_hat, {
            "method": "bisection_maxit",
            "expanded": n_expand,
            "iters": maxit,
            "bracket": (lo, hi),
            "f(lo)": flo,
            "f(hi)": fhi
        }

    # 3) Fallback: grid search minimize |W(p)-T|
    p_grid = np.linspace(p_lo, p_max_fallback, grid_fallback)
    vals = np.empty_like(p_grid)
    for i, p in enumerate(p_grid):
        vals[i] = abs(f(p))

    j = int(np.argmin(vals))
    p_best = float(p_grid[j])

    # Optional local refinement: try to bracket around p_best using neighbors
    # (works if f changes sign somewhere nearby)
    j_lo = max(j - 1, 0)
    j_hi = min(j + 1, len(p_grid) - 1)
    lo2, hi2 = float(p_grid[j_lo]), float(p_grid[j_hi])
    flo2, fhi2 = f(lo2), f(hi2)

    if flo2 * fhi2 <= 0 and hi2 > lo2:
        # small bisection in the local bracket
        lo, hi, flo, fhi = lo2, hi2, flo2, fhi2
        for it in range(maxit):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < tol:
                p_hat = mid
                return p_hat, 1.0 / p_hat, {
                    "method": "grid+local_bisection",
                    "expanded": n_expand,
                    "iters": it + 1,
                    "grid_best": p_best,
                    "local_bracket": (lo2, hi2),
                    "abs_error_grid_best": float(vals[j]),
                }
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        p_hat = 0.5 * (lo + hi)
        return p_hat, 1.0 / p_hat, {
            "method": "grid+local_bisection_maxit",
            "expanded": n_expand,
            "iters": maxit,
            "grid_best": p_best,
            "local_bracket": (lo2, hi2),
            "abs_error_grid_best": float(vals[j]),
        }

    # If even local bracketing fails, just return grid minimizer
    return p_best, 1.0 / p_best, {
        "method": "grid_only",
        "expanded": n_expand,
        "iters": 0,
        "grid_best": p_best,
        "abs_error_grid_best": float(vals[j]),
        "note": "No sign-change found; returning minimizer of |W(p)-T| on grid."
    }

def estimate_H_from_crossing(H_grid, W_vals):
    # Solve W(H) = 1 by linear interpolation on log-scale (more stable)
    y = np.log(W_vals)
    target = 0.0  # log(1)
    s = y - target

    idx = np.where(np.sign(s[:-1]) * np.sign(s[1:]) <= 0)[0]
    if len(idx) == 0:
        # fallback: closest point to 1
        return H_grid[np.argmin(np.abs(W_vals - 1.0))]

    i = idx[0]
    # interpolate between (H_i, y_i) and (H_{i+1}, y_{i+1})
    H0, H1 = H_grid[i], H_grid[i+1]
    y0, y1 = y[i], y[i+1]
    return H0 + (0.0 - y0) * (H1 - H0) / (y1 - y0)

def compute_W_curve(fbm_path, H_grid, L, K, T):
    W = np.array([normalized_p_variation(L, K, 1.0/h, T, fbm_path) for h in H_grid], dtype=float)
    return W

def monte_carlo_H_estimation(H, n_paths, L, K, T):
    print(f"Simulating {n_paths} paths for H={H}...")
    H_hat_list = []
    if H < 0.4:
        p_hi = 20.0
    elif H < 0.7:
        p_hi = 10.0
    else:
        p_hi = 5.0
        
    for _ in range(n_paths):
        fbm_path = davies_harte(T, L, H)
        p_hat, H_hat = estimate_p_hat(fbm_path, L, K, T, p_hi=p_hi)
        H_hat_list.append(H_hat)
    print(f"Done simulating H={H}.")
    return np.array(H_hat_list)

def divisors(n):
    ds = []
    for k in range(1, int(np.sqrt(n)) + 1):
        if n % k == 0:
            ds.append(k)
            if k != n // k:
                ds.append(n // k)
    return sorted(ds)

def log_spaced_divisors(n, m=120):
    cand = np.unique(np.round(np.logspace(0, np.log10(n), m)).astype(int))
    ds = [k for k in cand if k >= 2 and n % k == 0]
    return sorted(set(ds))

def estimate_p_hat_robust_2(X, L, K, T=1.0, p_lo=1.01, p_hi=20.0,maxit=80):
    """
    Robustly solve W(L,K,p,T,X)=T for p.

    - First tries bisection with an automatically expanded bracket.
    - If no sign change is found, falls back to minimizing |W(p)-T| on a grid,
      then refines locally with a small bisection around the best grid point if possible.

    Returns (p_hat, H_hat, info_dict).
    """
    def roots_in_interval_scipy(f, a, b, n_samples=2000, xtol=1e-12, rtol=1e-12, maxiter=200, merge_eps=1e-7):
        """
        Find (approximately) all real roots of f(x)=0 on [a,b] .
        """
        if a >= b:
            raise ValueError("Require a < b")
    
        xs = np.linspace(a, b, n_samples + 1)
        fs = np.array([f(x) for x in xs], dtype=float)
    
        finite = np.isfinite(fs)
        roots = []
    
        # If grid lands very close to a root
        near = np.where(finite & (np.abs(fs) < 10 * xtol))[0]
        roots.extend(xs[near].tolist())
    
        # Brackets from sign changes
        for i in range(len(xs) - 1):
            if not (finite[i] and finite[i + 1]):
                continue
            f1, f2 = fs[i], fs[i + 1]
            if f1 == 0.0:
                roots.append(xs[i])
            elif f1 * f2 < 0:
                sol = opt.root_scalar(
                    f,
                    bracket=(xs[i], xs[i + 1]),
                    method="brentq",
                    xtol=xtol,
                    rtol=rtol,
                    maxiter=maxiter,
                )
                if sol.converged:
                    roots.append(sol.root)
    
        # Merge duplicates
        roots = sorted(roots)
        merged = []
        for r in roots:
            if not merged or abs(r - merged[-1]) > merge_eps:
                merged.append(float(r))
            else:
                merged[-1] = 0.5 * (merged[-1] + r)
    
        return [r for r in merged if a - rtol <= r <= b + rtol]
    

    def W(p):
        return normalized_p_variation(L, K, p, T, X)

    def f(p):
        return W(p) - T
    

    root = np.array(roots_in_interval_scipy(f, p_lo, p_hi, maxiter=maxit))
    return root, 1.0/root