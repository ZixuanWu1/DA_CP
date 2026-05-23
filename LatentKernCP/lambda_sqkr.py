#######################################################################
### Adapted from "Quantile Regression in Reproducing Kernel Hilbert Space (Li, 2007)"
### Author:
### Date:    03/22/2025
#######################################################################
import numpy as np
from LatentKernCP.utils import kernel, pinball
from cvxopt import matrix, solvers

def _as_np_unique_sorted(a):
    a = np.asarray(a, dtype=int).ravel()
    if a.size == 0: return a
    return np.unique(a)

def lambda_path(S_vec, Phi, K, alpha, 
                max_steps=1000, eps = 1e-3, tol=1e-6, thres=10.0, ridge=1e-8, verbose=False):
    S_vec = np.asarray(S_vec, dtype=float).ravel()
    n = S_vec.size
    K = np.asarray(K, dtype=float)

    if Phi is not None:
        Phi = np.asarray(Phi, dtype=float)
        d = Phi.shape[1]
    else:
        Phi = np.ones((n, 1), dtype=float)
        d = 1

    # ----- Init -----
    ini = lambda_init(S_vec, Phi, K, alpha)
    indE = _as_np_unique_sorted(ini["indE"])
    indL = _as_np_unique_sorted(ini["indL"])
    indR = _as_np_unique_sorted(ini["indR"])
    lam  = float(ini["lambda"])
    v    = np.asarray(ini["v"], dtype=float).copy()
    eta  = np.asarray(ini["eta"], dtype=float).copy() if d > 0 else float(ini["eta"])

    if verbose:
        print(f"[init] lambda={lam:.6g}; |E|={indE.size}, |L|={indL.size}, |R|={indR.size}")

    # storage
    eta_arr = np.zeros((max_steps+1, d))
    v_arr   = np.zeros((max_steps+1, n))
    Csic    = np.zeros(max_steps+1)
    fit     = np.zeros((max_steps+1, n))
    lambda_vals = np.zeros(max_steps+1)
    Elbows  = [None]*(max_steps+1)

    eta_arr[0] = eta
    v_arr[0] = v
    lambda_vals[0] = lam
    Elbows[0] = indE.copy()

    # current fit
    Kv = K @ v
    g_hat = Phi @ eta + Kv / lam

    csic = np.log(pinball(g_hat, S_vec, 1-alpha) / n) + (np.log(n)/(2*n)) * indE.size
    Csic[0] = csic
    fit[0] = g_hat

    k = 0
    while k < max_steps:
        k += 1
        if indL.size == 0 and indR.size == 0:
            if verbose: print("[stop] both L and R empty")
            break

        # ----- Step 1: Solve linear system for lambda step size -----
        # Solve [[PhiE, KEE], [0_{dxd}, PhiE^T]] [b_eta; b_v] = [S_E; 0_d]
        E = indE.copy()
        m = E.size
        KEE = K[np.ix_(E, E)]
        PhiE = Phi[E, :]
        
        A_top = np.hstack((PhiE, KEE))
        A_bot = np.hstack((np.zeros((d, d)), PhiE.T))
        A = np.vstack((A_top, A_bot))
        b = np.concatenate((S_vec[E], np.zeros(d)))
        
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, b, rcond=None)[0]
        
        b_eta = delta[:d]
        b_v   = delta[d:]

        h_l = Phi @ b_eta + K[:, E] @ b_v  # (n, )

        # ----- Step 2: Find the next lambda and event -----
        cand_steps, cand_who, cand_dir = [], [], []

        # (a) point in E hits a bound: v_E -> -alpha or 1-alpha
        if m > 0:
            vE = v[E].astype(float)
            bv = np.asarray(b_v, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                step_to_lo = (-alpha      - vE) / bv
                step_to_hi = (1.0 - alpha - vE) / bv
            for loc in range(m):
                if not np.isfinite(bv[loc]) or np.abs(bv[loc]) <= tol:
                    continue
                s1 = step_to_lo[loc]
                s2 = step_to_hi[loc]
                s_candidates = [s for s in (s1, s2) if np.isfinite(s) and s < -tol]
                if not s_candidates:
                    continue
                t = max(s_candidates)                    # largest negative = closest to 0
                dir_flag = -1 if abs(t - s1) < abs(t - s2) else +1
                cand_steps.append(float(t))
                cand_who.append(('leave', int(E[loc])))
                cand_dir.append(dir_flag)

        # (b) residual zero for i in L∪R (indices are in notE by construction)
        LR = np.concatenate((indL, indR))
        denom = S_vec[LR] - h_l[LR]
        safe = np.abs(denom) > 1e-12
        if np.any(safe):
            ratio = np.full(LR.size, np.nan, dtype=float)
            ratio[safe] = (g_hat[LR][safe] - h_l[LR][safe]) / denom[safe]
            good = safe & np.isfinite(ratio) & (ratio <= 1.0)
            if np.any(good):
                steps = lam * (ratio[good] - 1.0)
                mask  = steps < -tol
                if np.any(mask):
                    for idx_i, t in zip(LR[good][mask], steps[mask]):
                        dir_flag = -1 if idx_i in indL else 1
                        cand_steps.append(float(t))
                        cand_who.append(('hit', int(idx_i)))
                        cand_dir.append(dir_flag)

        if not cand_steps:
            if verbose: print("[stop] no more candidate events")
            break

        cand_steps = np.asarray(cand_steps, float)
        imax = int(np.argmax(cand_steps))  # largest negative (closest to 0)
        step = float(cand_steps[imax])
        ev_kind, ev_idx = cand_who[imax]
        ev_dir = cand_dir[imax]

        if verbose:
            print(f"[{k}] event={ev_kind} idx={ev_idx} step={step:.6g} dir={ev_dir}")

        # take step
        lam_next = lam + step
        if verbose:
            print(f"[{k}] lambda_next={lam_next:.6g}")

        # early stops
        if lam_next <= 0 or not np.isfinite(lam_next):
            if verbose: print("[stop] lambda hit nonpositive")
            break

        if np.abs(lam - lam_next) < eps and lam_next < thres:
            if verbose: print("[stop] descent too small")
            break
            
        # ----- Step 3: Update the next E, L, R -----
        if ev_kind == 'leave':
            indE = indE[indE != ev_idx]
            if ev_dir == -1:
                indL = _as_np_unique_sorted(np.append(indL, ev_idx))
            else:
                indR = _as_np_unique_sorted(np.append(indR, ev_idx))
        else:
            if ev_idx in indL:
                indL = indL[indL != ev_idx]
            elif ev_idx in indR:
                indR = indR[indR != ev_idx]
            indE = _as_np_unique_sorted(np.append(indE, ev_idx))

        if len(indE) == 0:
            if verbose: print("[stop] E is empty")
            break

        # ----- Step 4: Update v and eta -----
        v = np.zeros(n, dtype=float)
        v[indL] = -alpha
        v[indR] = 1 - alpha

        E = indE.copy()
        m = len(indE)

        KEE = K[np.ix_(E, E)]
        KEL, KER = K[np.ix_(E, indL)], K[np.ix_(E, indR)]
        PhiE = Phi[E, :]
        PhiL, PhiR = Phi[indL, :], Phi[indR, :]
        one_L = np.ones(len(indL))
        one_R = np.ones(len(indR))

        S_E  = np.asarray(S_vec[E],  float).ravel()
        S_E_eff = S_E - (-alpha * (KEL @ one_L) + (1 - alpha) * (KER @ one_R)) / lam_next
        b_bot = alpha * (PhiL.T @ one_L) - (1 - alpha) * (PhiR.T @ one_R)

        # Build the stacked linear operator H z ≈ b  (least-squares)
        # H = [[PhiE, KEE], [0, PhiE^T]], b = [S_E, 0]
        top = np.hstack([PhiE, KEE / lam_next])
        bot = np.hstack([np.zeros((d, d)), PhiE.T])
        H   = np.vstack([top, bot])
        b   = np.concatenate([S_E_eff, b_bot])
        P_np = H.T @ H + ridge * np.eye(d + m)
        q_np = -(H.T @ b)

        # Inequalities: box on v_E only
        #  v_E ≤ (1-α) → [0_dxm  I_m] z ≤ (1-α)·1
        # -v_E ≤ α     → [0_dxm -I_m] z ≤ α·1
        Zdm = np.zeros((m, d))
        G_np = np.vstack([
            np.hstack([Zdm,  np.eye(m)]),
            np.hstack([Zdm, -np.eye(m)])
        ])
        h_np = np.hstack([
            (1.0 - alpha) * np.ones(m),
            alpha * np.ones(m)
        ])

        # CVXOPT solve
        solvers.options['show_progress'] = False
        P = matrix(P_np, tc='d'); q = matrix(q_np, tc='d')
        G = matrix(G_np, tc='d'); h = matrix(h_np, tc='d')
        try:
            sol = solvers.qp(P, q, G, h)
            if sol['status'] != 'optimal':
                if verbose: print("Warning: QP status:", sol['status'])
                z = np.linalg.lstsq(H, b, rcond=1e-12)[0]
            else:
                z = np.array(sol['x']).flatten()
        except Exception as e:
            if verbose: print("QP exception:", repr(e))
            z = np.linalg.lstsq(H, b, rcond=1e-12)[0]

        eta = z[:d]
        vE  = z[d:]
        v[E] = vE

        # update fit
        lam = lam_next
        Kv = K @ v
        g_hat = Phi @ eta + Kv / lam

        # store
        fit[k] = g_hat
        lambda_vals[k] = lam
        v_arr[k]    = v
        eta_arr[k]  = eta
        Elbows[k]   = E
        Csic[k]     = np.log(pinball(g_hat, S_vec, 1-alpha) / n) + (np.log(n)/(2*n)) * len(E)

    lambda_opt = lambda_vals[np.argmin(Csic[:k])]

    if verbose:
        print(f"[done] steps={k}, |E|={indE.size}, lambda_opt={lambda_opt:.6g}")

    return {
        "lambdas": lambda_vals[:k],
        "v_arr": v_arr[:k],
        "eta_arr": eta_arr[:k],
        "Elbows": Elbows[:k],
        "fit": fit[:k],
        "Csic": Csic[:k],
        "steps": k
    }


def lambda_init(S_vec, Phi, K, alpha, verbose=False):
    n = len(S_vec)
    d = Phi.shape[1] if Phi is not None else 0

    quant = np.quantile(S_vec, 1 - alpha, method="higher")
    istar = int(np.argmin(np.abs(S_vec - quant)))
    S_star = S_vec[istar]

    # Find the next point entering elbow
    notindE = np.setdiff1d(np.arange(n), np.array([istar]), assume_unique=True)
    if Phi is not None:
        k_star = np.argmax(np.abs(Phi[istar,]))
        eta_k_star = S_star / Phi[istar, k_star]

        # Initialize sets
        g_hat = Phi[:,k_star] * eta_k_star
        indE = [istar]
        indR = np.where(S_vec > g_hat)[0].tolist()
        indL = np.where(S_vec < g_hat)[0].tolist()

        v_star = (alpha*Phi[notindE, k_star].sum() - Phi[indR,k_star].sum())/Phi[istar,k_star]
    else:
        k_star = 0

        # Initialize sets
        indE = [istar]
        indR = np.where(S_vec > S_vec[istar])[0].tolist()
        indL = np.where(S_vec < S_vec[istar])[0].tolist()

        v_star = alpha*len(indL)-(1-alpha)*len(indR)

    # Update kernel variables
    eps_bnd = 1e-04
    v_star_clipped = np.clip(v_star, -alpha + eps_bnd, 1-alpha - eps_bnd)
    if verbose: print(f"v_star: {v_star}, clipped: {v_star_clipped}")

    v_init = np.full(n, -alpha, dtype=float)
    v_init[istar] = v_star_clipped
    v_init[indR] = 1 - alpha
    
    # Find initial lambda
    f_hat = K @ v_init
    f_hat_star = g_hat[istar]

    if Phi is not None:
        denom_all = S_vec[notindE] - S_star * (Phi[notindE, k_star] / Phi[istar,k_star])
        num_all = f_hat[notindE] - f_hat_star * (Phi[notindE, k_star] / Phi[istar,k_star])
    else:
        denom_all = S_vec[notindE] - S_star
        num_all = g_hat[notindE] - f_hat_star

    valid = np.where(denom_all != 0)[0]
    if valid.size == 0:
        lam = np.inf
    else:
        cands = num_all[valid] / denom_all[valid]
        pos = cands[cands > 0]
        lam = np.max(pos) if pos.size > 0 else np.inf

    # Update linear variables
    if Phi is not None:
        if not np.isfinite(lam) or lam == 0:
            eta_init = np.zeros(d)
        else:
            eta_init = np.zeros(d)
            eta_init[k_star] = (S_star - (1.0 / lam) * f_hat_star) / Phi[istar,k_star]
    else:
        eta_init = S_star - f_hat_star if np.isfinite(lam) and lam != 0 else 0.0

    return {
        "v": v_init,
        "eta": eta_init,
        "lambda": lam,
        "indE": indE,
        "indR": indR,
        "indL": indL
    }