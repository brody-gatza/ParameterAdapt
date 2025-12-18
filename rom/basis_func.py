import numpy as np

def adapt_basis(solver_param,rom_param,Q_bar_new_solver_int):

    q_ref                  = rom_param['q_ref']
    normalizor             = rom_param['norm']
    rom_param['old_basis'] = rom_param['basis']

    if solver_param['adaptive_rom_method'] == 'direct':

        # roll the training window to the left
        F   = np.roll(rom_param['F']   , shift=-1,axis=1)
        Q_R = np.roll(rom_param['Q_R'] , shift=-1,axis=1)
        
        # include the estimated snapshot
        F [:,-1]          = (Q_bar_new_solver_int-q_ref)*normalizor
        Q_R[:,-1]         = rom_param['basis'].T @ F [:,-1]

        # solve the basis adaptation minimization problem
        pinv_Q_R = np.linalg.pinv(Q_R)
        rom_param['basis'] = F @ pinv_Q_R

        #orthogonalize the basis
        rom_param['basis'] , _ , _ = np.linalg.svd(rom_param['basis']
                                                   ,full_matrices=False)

        # update the states
        rom_param['F']      = F
        rom_param['Q_R']    = Q_R

    if solver_param['adaptive_rom_method'] == 'ojas':

        V_old = rom_param['old_basis']
        eta   = 1
        u_new = (Q_bar_new_solver_int-q_ref)*normalizor
        orthornomalize = True

        # Oja update
        V_new = rom_param['old_basis'] + eta * np.outer(u_new, u_new.T @ V_old)
    
        # Orthonormalize if desired
        if orthornomalize:
            V_new, _ = np.linalg.qr(V_new, mode="reduced")

        rom_param['basis'] = V_new

    if solver_param['adaptive_rom_method'] == 'grouse':

        eta=0.1 
        eps=1e-10 
        orthonormalize=True
        V_old = rom_param['basis']
        u_new = (Q_bar_new_solver_int-q_ref)*normalizor
        u_new = u_new[:,None]

        """
        GROUSE (Grassmannian Rank-One Update Subspace Estimation) basis update.
        
        Inputs
        ------
        V_old : (N,K) ndarray
            Current basis.
        u_new : (N,1) ndarray
            New snapshot to adapt the basis with.
        eta : float
            Step size parameter.
        eps : float
            Tolerance for numerical stability.
        orthonormalize : bool
            Whether to orthonormalize the updated basis.
        
        Returns
        -------
        V_new : (N,K) ndarray
            Updated basis.
        """
        # weights for projection of u_new onto current subspace
        w = V_old.T @ u_new  # shape (K,)
        p = V_old @ w  # projection of u_new onto span(V_old)
        r = u_new - p  # residual
        
        w_norm = np.linalg.norm(w)
        p_norm = np.linalg.norm(p)
        r_norm = np.linalg.norm(r)
        
        if w_norm < eps or p_norm < eps or r_norm < eps:
            V_new = V_old.copy()
        else:
            # GROUSE step size parameter
            sigma = r_norm * p_norm
            
            # angle for the geodesic step on the Grassmannian
            alpha = eta * sigma
            
            # direction in the 2D plane spanned by p and r
            y = (np.cos(alpha) - 1.0) * (p / p_norm) + np.sin(alpha) * (r / r_norm)
            
            # rank-one Grassmannian update
            V_new = V_old + (y @ w.T / w_norm)
        
        # Orthonormalize if desired
        if orthonormalize:
            V_new, _ = np.linalg.qr(V_new, mode="reduced")
        
        rom_param['basis'] = V_new

    if solver_param['adaptive_rom_method'] == 'isvd':

        forgetting_factor=0.0 
        r=None
        tol=1e-12 
        orthonormalize=True
        V_old = rom_param['basis']
        S_old = rom_param['isvd_singular']
        u_new = (Q_bar_new_solver_int-q_ref)*normalizor

        """
        Incremental-SVD basis update.
        
        Inputs
        ------
        V_old : (N,K) ndarray
            Current basis.
        S_old : (K,) ndarray
            Current singular values associated with V_old.
        u_new : (N,1) ndarray
            New snapshot to adapt the basis with.
        forgetting_factor : float in [0,1]
            Amount of “forgetting” applied to the old singular values (1 = none, 0 = fully forget).
        r : int or None
            Target rank after update.
        tol : float
            Tolerance to skip adaptation if residual is small.
        orthonormalize : bool
            Whether to orthonormalize the updated basis.

        Returns
        -------
        V_new : (N,K) ndarray
            Updated basis (truncated to target rank).
        S_new : (K,) ndarray
            Updated singular values (truncated to target rank).
        """

        # Initialize
        U = V_old
        s = S_old.ravel()
        Kdim = U.shape[1]
        if r is None:
            r = Kdim

        # Project and residual
        y = u_new                           # new snapshot (N,1)
        p = U.T @ y                         # reduced coords of new snapshot in old basis (K,1)
        q = y - (U @ p)                     # residual (N,1)
        qnorm = float(np.linalg.norm(q))    # residual norm

        # Check if we have new information
        if qnorm > tol:
            # New orthogonal direction
            u_perp = q / qnorm # (N,1)

            # Build small (K+1)x(K+1) core matrix
            Kcore = np.zeros((Kdim + 1, Kdim + 1), dtype=U.dtype)
            Kcore[:Kdim, :Kdim] = np.diag(forgetting_factor * s[:Kdim]) # apply forgetting
            Kcore[:Kdim,  Kdim] = p.ravel()
            Kcore[ Kdim,  Kdim] = qnorm

            # SVD of the core matrix
            Ubar, s_new, _ = np.linalg.svd(Kcore, full_matrices=False)

            # Augment U with u_perp, then rotate by Ubar and truncate
            U_aug = np.hstack([U, u_perp[:,np.newaxis]])                  # (N, K+1)
            U_new_full = U_aug @ Ubar                       # (N, K+1)

            k_new = min(r, U_new_full.shape[1])
            V_new = U_new_full[:, :k_new]
            S_new = s_new[:k_new].copy()
        else:
            # No new information; just apply forgetting
            k_new = min(r, Kdim)
            V_new = U[:, :k_new].copy()
            S_new = (forgetting_factor * s[:k_new]).copy()

        # Check if orthonormalization is desired
        if orthonormalize:
            V_new, _ = np.linalg.qr(V_new, mode="reduced")

        rom_param['basis']          = V_new
        rom_param['isvd_singular']  = S_new

    if solver_param['adaptive_rom_method'] == 'past':

        V_old = rom_param['basis']
        P_old = rom_param['past_rls']
        u_new = (Q_bar_new_solver_int-q_ref)*normalizor
        u_new = u_new[:,None]
        lam   = 1.0 
        orthonormalize=True

        """
        PAST (Projection Approximation Subspace Tracking) basis update.
        
        Inputs
        ------
        V_old : (N,K) ndarray
            Current basis.
        P_old : (K,K) ndarray
            Inverse correlation (RLS) matrix for PAST recursion.
        u_new : (N,1) ndarray
            New snapshot to adapt the basis with.
        lam : float in (0,1]
            Forgetting factor (lam=1 -> no forgetting).
        orthonormalize : bool
            Whether to orthonormalize the updated basis.
        
        Returns
        -------
        V_new : (N,K) ndarray
            Updated basis.
        P_new : (K,K) ndarray
            Updated RLS matrix for PAST.
        """
        # u_new = np.asarray(u_new).ravel()  # ensure (N,)
        
        w = V_old.T @ u_new # projection coefficients
        beta = 1 + (w.T @ P_old @ w) / lam  # denominator
        v = (P_old @ w) / lam  # intermediate vector
        P_new = P_old / lam - np.outer(v, v) / beta  # RLS matrix update
        V_new = V_old + (u_new - V_old @ w) @ (P_new @ w).T   # basis update
        
        # Orthonormalize if desired
        if orthonormalize:
            V_new, _ = np.linalg.qr(V_new, mode="reduced")
    
        rom_param['basis']          = V_new
        rom_param['past_rls']       = P_new

    return rom_param
