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

    return rom_param