import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel
from LatentKernCP.utils import kernel, pinball

def _as_np_unique_sorted(a):
    a = np.asarray(a, dtype=int).ravel()
    if a.size == 0: return a
    return np.unique(a)

import numpy as np
from cvxopt import matrix, solvers

def S_path(
    S_cal, Phi, K, lam, alpha, alpha0, best_v,
    start_side='left', max_steps=200, eps=1e-3, tol=1e-6, ridge=1e-8, verbose=False
):
    """
    Piecewise-linear path in S, maintaining elbow sets (E/L/R) and solving
    the constrained LS/QP for (eta, v_E) at each event.

    Expects helpers defined elsewhere:
      - S_init(S_cal, Phi, K, lam, alpha, best_v, start_side, eps, tol)
      - _as_np_unique_sorted(arr)
      - pinball(g_hat, S_vec, q)
    """

    # ---------- init & shapes ----------
    S_cal = np.asarray(S_cal, dtype=float).ravel()
    n = S_cal.size + 1
    K = np.asarray(K, dtype=float)

    if Phi is not None:
        Phi = np.asarray(Phi, dtype=float)
        d = int(Phi.shape[1])
    else:
        Phi = np.ones((n, 1), dtype=float)
        d = 1

    ini   = S_init(S_cal, Phi, K, lam, alpha, best_v, start_side, eps, tol)
    indE  = _as_np_unique_sorted(ini["indE"])
    indL  = _as_np_unique_sorted(ini["indL"])
    indR  = _as_np_unique_sorted(ini["indR"])
    S     = float(ini["S"])
    v     = np.asarray(ini["v"],   dtype=float).copy()
    eta   = np.asarray(ini["eta"], dtype=float).copy()
    S_ini = S

    if verbose:
        print(f"[init] S={S:.6g}; |E|={indE.size}, |L|={indL.size}, |R|={indR.size}")

    # ---------- storage ----------
    eta_arr = np.zeros((max_steps+1, d))
    v_arr   = np.zeros((max_steps+1, n))
    Csic    = np.zeros(max_steps+1)
    fit     = np.zeros((max_steps+1, n))
    S_vals  = np.zeros(max_steps+1)
    Elbows  = [None]*(max_steps+1)

    eta_arr[0] = eta
    v_arr[0]  = v
    S_vals[0] = S
    Elbows[0] = indE.copy()

    # current fit/residuals
    Kv   = K @ v
    g_hat = Phi @ eta + Kv / lam
    S_vec = np.append(S_cal, S)
    r     = S_vec - g_hat

    Csic[0] = np.log(pinball(g_hat, S_vec, 1-alpha) / n) + (np.log(n) / (2 * n)) * indE.size
    fit[0]  = g_hat

    # ---------- main loop ----------
    k = 0
    # cache only A_dag keyed by tuple(E)
    A_cache = {}

    while k < max_steps:
        k += 1
        # stop if no active box constraints remain OR test already in R
        if (indL.size == 0 and indR.size == 0):
            if verbose:
                print("[stop] L and R empty")
            break

        # ----- Step 1: derivatives wrt S via elbow system -----
        E = indE
        m = E.size

        if m == 0:
            dvE_dS = np.zeros(0)
            dg_dS  = np.zeros(n)
            dr_dS  = -dg_dS
            dr_dS[n-1] += 1.0
        else:
            KEE = K[np.ix_(E, E)]
            PhiE = Phi[E, :]
            GPhi = PhiE.T @ PhiE + ridge * np.eye(d)
            FEpinv = np.linalg.inv(GPhi)
            PiE = np.eye(m) - PhiE @ FEpinv @ PhiE.T

            key = tuple(E.tolist())
            if key in A_cache:
                A_dag = A_cache[key]
            else:
                A = PiE @ KEE @ PiE
                A_dag = np.linalg.pinv(A, rcond=1e-12)
                A_cache[key] = A_dag

            e_test = np.zeros(m)
            e_test[E == (n-1)] = 1.0

            dvE_dS = lam * (A_dag @ (PiE @ e_test))
            rhs_deta = PhiE.T @ (e_test - (KEE @ dvE_dS) / lam)
            deta_dS  = np.linalg.solve(GPhi, rhs_deta)
            dg_dS    = Phi @ deta_dS + (K[:, E] @ dvE_dS) / lam

            dr_dS = -dg_dS
            dr_dS[n-1] += 1.0

        # ----- Step 2: find next event (smallest positive step) -----
        cand_steps, cand_who, cand_dir = [], [], []

        # (a) elbow variable hits a bound
        if m > 0:
            vE_now = v[E].astype(float)
            for loc in range(m):
                slope = float(dvE_dS[loc])
                if abs(slope) <= tol:
                    continue
                t1 = (-alpha        - vE_now[loc]) / slope  # to -alpha
                t2 = ((1.0 - alpha) - vE_now[loc]) / slope  # to 1-alpha
                t_candidates = [t for t in (t1, t2) if t > tol]
                if not t_candidates:
                    continue
                t = min(t_candidates)
                # dir used to dispatch into L/R later
                dir_flag = -1 if abs(t - t1) < abs(t - t2) else +1
                cand_steps.append(t)
                cand_who.append(('leave', int(E[loc])))
                cand_dir.append(dir_flag)

        # (b) residual in L ∪ R hits zero
        LR = np.concatenate((indL, indR))
        for i in LR:
            slope = float(dr_dS[i])
            if abs(slope) <= tol:
                continue
            t = - float(r[i]) / slope  # correct step formula
            if t > tol:
                cand_steps.append(t)
                cand_who.append(('hit', int(i)))
                cand_dir.append(int(np.sign(slope)))

        if not cand_steps:
            if verbose: print("[stop] no more candidate events")
            break

        cand_steps = np.asarray(cand_steps, float)
        imin = int(np.argmin(cand_steps))
        step = float(cand_steps[imin])
        ev_kind, ev_idx = cand_who[imin]
        ev_dir = cand_dir[imin]

        # If the test index leaves the elbow, finalize state and stop
        if ev_idx == n-1:
            if verbose: 
                print(f"[stop] exited elbow at S={S:.6g}")
            break

        if step <= tol:
            if verbose:
                print(f"[stop] step too small ({step:.3e})")
            break

        if S + step > np.max(S_cal) + eps:
            if verbose:
                print(f"[stop] exceeded max S_cal + eps, S={S + step:.6g}")
            break

        S_next = S + step
        if verbose:
            print(f"[{k}] event={ev_kind}, idx={ev_idx}, step={step:.6g} -> S={S_next:.6g}, dir={ev_dir}")


        # ----- Step 3: update sets E/L/R based on event -----
        if ev_kind == 'leave':
            # variable leaves E to a box side
            E_new = E[E != ev_idx]
            if ev_dir < 0:
                indL = _as_np_unique_sorted(np.append(indL, ev_idx))
            else:
                indR = _as_np_unique_sorted(np.append(indR, ev_idx))
            indE = _as_np_unique_sorted(E_new)
        else:  # 'hit' -> constraint from L or R becomes active -> move into E
            if ev_idx in indL:
                indL = indL[indL != ev_idx]
            else:
                indR = indR[indR != ev_idx]
            indE = _as_np_unique_sorted(np.append(indE, ev_idx))


        # ----- Step 4: solve for (eta, v_E) at S_next -----
        # set box values for L/R
        v = np.zeros(n, dtype=float)
        v[indL] = -alpha
        v[indR] = 1 - alpha

        E = indE.copy()
        m = len(E)

        KEE = K[np.ix_(E, E)]
        KEL, KER = K[np.ix_(E, indL)], K[np.ix_(E, indR)]
        PhiE = Phi[E, :]
        PhiL, PhiR = Phi[indL, :], Phi[indR, :]
        one_L = np.ones(len(indL))
        one_R = np.ones(len(indR))

        S_vec = np.append(S_cal, S_next)
        S_E  = np.asarray(S_vec[E], float).ravel()
        S_E_eff = S_E - (-alpha * (KEL @ one_L) + (1 - alpha) * (KER @ one_R)) / lam
        b_bot = alpha * (PhiL.T @ one_L) - (1 - alpha) * (PhiR.T @ one_R)

        # Build QP: minimize 0.5 z^T P z + q^T z, subject to box on v_E
        top = np.hstack([PhiE, KEE / lam])                   # [m, d+m]
        bot = np.hstack([np.zeros((d, d)), PhiE.T])          # [d, d+m]
        H   = np.vstack([top, bot])                          # [(m+d) x (d+m)]
        b   = np.concatenate([S_E_eff, b_bot])
        P_np = H.T @ H + ridge * np.eye(d + m)
        q_np = -(H.T @ b)

        Zdm = np.zeros((m, d))
        G_np = np.vstack([
            np.hstack([Zdm,  np.eye(m)]),   # v_E <= 1 - alpha
            np.hstack([Zdm, -np.eye(m)])    # -v_E <= alpha
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

        # advance S and recompute state
        S = S_next
        Kv   = K @ v
        g_hat = Phi @ eta + Kv / lam
        S_vec = np.append(S_cal, S)
        r     = S_vec - g_hat

        # store
        fit[k]    = g_hat
        S_vals[k] = S
        v_arr[k]  = v
        eta_arr[k] = eta
        Elbows[k] = indE.copy()
        Csic[k]   = np.log(pinball(g_hat, S_vec, 1-alpha) / n) + (np.log(n)/(2*n))*indE.size

        # Break when v_n+1 is bigger than threshold
        if vE[-1] > alpha0:
            if verbose: print(f"[stop] exceeded randomized threshold, v_n+1={vE[-1]:.6g}")
            break

    if verbose:
        print(f"[done] steps={k}, |E|={indE.size}, S={S:.6g}")

    return {
        "S_vals":  S_vals[:k],
        "v_arr":   v_arr[:k],
        "eta_arr": eta_arr[:k],
        "Elbows":  Elbows[:k],
        "fit":     fit[:k],
        "Csic":    Csic[:k],
        "steps":   k,
        "S_opt":   S,
        "S_init":  S_ini
    }


def S_init(S_cal, Phi, K, lam, alpha, opt_v_lambda,
               start_side='left', eps=1e-1, tol=1e-6, ridge=1e-8, verbose=False):
    
    # Initialize sets
    indE = np.where((opt_v_lambda > -alpha + tol) & (opt_v_lambda < 1.0 - alpha - tol))[0].tolist()
    indL = np.where(np.abs(opt_v_lambda + alpha) <= tol)[0].tolist()
    indR = np.where(np.abs(opt_v_lambda - (1.0 - alpha)) <= tol)[0].tolist()

    n = len(S_cal)
    d = Phi.shape[1]
    E = indE.copy()
    m = len(E)

    v = np.zeros(n, dtype=float)
    v[indL] = -alpha
    v[indR] = 1 - alpha

    alpha0 = -alpha if start_side == 'left' else 1-alpha

    KEE, kEn = K[np.ix_(E, E)], K[E, n]
    KEL, KER = K[np.ix_(E, indL)], K[np.ix_(E, indR)]
    PhiE, Phin = Phi[E, :], Phi[n]
    PhiL, PhiR = Phi[indL, :], Phi[indR, :]
    one_L = np.ones(len(indL))
    one_R = np.ones(len(indR))

    # Top (correct signs)
    S_E = S_cal[indE]
    S_E_eff = S_E - (alpha0 * kEn - alpha * (KEL @ one_L) + (1 - alpha) * (KER @ one_R)) / lam
    top = np.hstack([PhiE, KEE / lam])
    b_top = S_E_eff

    # Bottom (transpose + sign)
    bot = np.hstack([np.zeros((d, d)), PhiE.T])
    b_bot = -alpha0 * Phin + alpha * (PhiL.T @ one_L) - (1 - alpha) * (PhiR.T @ one_R)

    H = np.vstack([top, bot])
    b = np.concatenate([b_top, b_bot])

    # QP matrices sized with m, not n
    # Inequality constraints: (quantile - 1) <= beta <= quantile
    # These translate to:
    # 1) beta <= 1-alpha           => I @ beta <= 1-alpha * ones(N)
    # 2) -alpha <= beta     => -I @ beta <= alpha * ones(N)
    P_np = H.T @ H + ridge * np.eye(d + m)
    q_np = -(H.T @ b)
    Zdm = np.zeros((m, d))
    G_np = np.vstack([
        np.hstack([Zdm,  np.eye(m)]),
        np.hstack([Zdm, -np.eye(m)])
    ])
    h_np = np.hstack([
        (1.0 - alpha) * np.ones(m),
        alpha * np.ones(m)
    ])

    # Convert to cvxopt types
    P = matrix(P_np, tc='d')
    q = matrix(q_np, tc='d')
    G = matrix(G_np, tc='d')
    h = matrix(h_np, tc='d')

    Aeq, beq = None, None
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, Aeq, beq)

    if sol['status'] == 'optimal':
        delta = np.array(sol['x']).flatten()
    else:
        print("Warning: CVXOPT did not find an optimal solution. Status:", sol['status'])

    z = np.array(sol['x']).ravel()
    eta_init = z[:d]
    delta  = z[d:]
    v[E] = delta

    v_init = np.append(v, alpha0)
    g_hat = Phi @ eta_init + (K @ v_init) / lam
    S_init = g_hat[-1]

    indE.append(n) # start with n+1 in indE
    
    return {
        "v": v_init,
        "eta": eta_init,
        "S": S_init,
        "indE": indE,
        "indR": indR,
        "indL": indL
    }