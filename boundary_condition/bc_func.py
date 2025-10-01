import numpy as np

from utils import reshape_func 


def update_ghost_cell(solver_param,state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']
    Q_cons   = state['Q_cons']
    Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

    if solver_param['gas_model'] != 'Air':

        num_eqn = 5

    else:

        num_eqn = 4

    for eqn in range(num_eqn):

        inlet_bc  = solver_param['bc_data'][eqn,0]
        outlet_bc = solver_param['bc_data'][eqn,1]

        ################################################## inlet ################################################## 
        
        ##### periodic ##### 

        if inlet_bc == 'periodic':

                Q_prim_user[eqn,0:2] = Q_prim_user[eqn,-3].reshape(-1,1)

                
        ##### wall ##### 

        elif inlet_bc == 'wall':
            
            # velocity
            if eqn == 1: 

                Q_prim_user[eqn,0:2] = - Q_prim_user[eqn,2].reshape(-1,1)

            else:

                Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

        ##### value/extrap ##### 

        else:
            
            ##### extrap #####

            if inlet_bc == 'extrapolate':

                Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

            ##### value ##### 

            else: 
                
                if eqn == 4:

                    if '/' in inlet_bc:
                        
                        parts = inlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[4:,0:2] = value + A*np.sin(f*t)

                    else:
                    
                        Q_prim_user[4:,0:2] = np.array(eval(inlet_bc)).reshape(-1,1)

                else:

                    if '/' in inlet_bc:
                        
                        parts = inlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[eqn,0:2] = value + A*np.sin(f*t)


                    else:

                        Q_prim_user[eqn,0:2] = float(inlet_bc)


        ################################################## outlet ################################################## 
        
        ##### periodic ##### 

        if outlet_bc == 'periodic':

            Q_prim_user[eqn,-2:] = Q_prim_user[eqn,2].reshape(-1,1)

        ##### wall ##### 

        elif outlet_bc == 'wall':

            # velocity
            if eqn == 1: 

                Q_prim_user[eqn,-2:] = - Q_prim_user[eqn,-3].reshape(-1,1)
                
            else:

                Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

        ##### value/extrap ##### 

        else:
            
            ##### extrap #####

            if outlet_bc == 'extrapolate':

                Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

            ##### value ##### 

            else:

                if eqn == 4:

                    if '/' in outlet_bc:
                        
                        parts = outlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[4:,-2:] = value + A*np.sin(f*t)

                    else:
                    
                        Q_prim_user[4:,-2:] = np.array(eval(outlet_bc)).reshape(-1,1)

                else:

                    if '/' in outlet_bc:
                        
                        parts = outlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[eqn,-2:] = value + A*np.sin(f*t)


                    else:
                    
                        Q_prim_user[eqn,-2:] = float(outlet_bc)


        # if outlet_bc == 'periodic':

        #     Q_prim_user[eqn,-2:] = Q_prim_user[eqn,0:2].reshape(-1,1)

        # ##### wall ##### 

        # if inlet_bc == 'wall':
            
        #     # velocity
        #     if eqn == 1: 

        #         Q_prim_user[eqn,0:2] = - Q_prim_user[eqn,2].reshape(-1,1)

        #     else:

        #         Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

        # if outlet_bc == 'wall':

        #     # velocity
        #     if eqn == 1: 

        #         Q_prim_user[eqn,-2:] = - Q_prim_user[eqn,-3].reshape(-1,1)
                
        #     else:

        #         Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

        # ##### inlet extrapolate ##### 

        # if inlet_bc == 'extrapolate':

        #     Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

        # else:
            
        #     if eqn == 4:

        #         if '/' in inlet_bc:
                    
        #             parts = inlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[4:,0:2] = value + A*np.sin(f*t)

        #         else:
                
        #             Q_prim_user[4:,0:2] = np.array(eval(inlet_bc)).reshape(-1,1)

        #     else:

        #         if '/' in inlet_bc:
                    
        #             parts = inlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[eqn,0:2] = value + A*np.sin(f*t)


        #         else:

        #             Q_prim_user[eqn,0:2] = float(inlet_bc)

        # ##### outlet extrapolate ##### 

        # if outlet_bc == 'extrapolate':

        #     Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

        # else:

        #     if eqn == 4:

        #         if '/' in outlet_bc:
                    
        #             parts = outlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[4:,-2:] = value + A*np.sin(f*t)

        #         else:
                
        #             Q_prim_user[4:,-2:] = np.array(eval(outlet_bc)).reshape(-1,1)

        #     else:

        #         if '/' in outlet_bc:
                    
        #             parts = outlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[eqn,-2:] = value + A*np.sin(f*t)


        #         else:
                 
        #             Q_prim_user[eqn,-2:] = float(outlet_bc)

    
    Q_prim_solver = reshape_func.results_user2solver_converter(Q_prim_user)
    
    state['Q_prim'] = Q_prim_solver

    return state