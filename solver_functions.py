import os
import numpy as np
import cantera as ct
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import time_integrator_functions
import rom_functions
import visualization_functions


def solver_parameters_collector(args,input_param):

    solver_param = {}

    ### solver_mode ###
    if input_param['solver_mode'] == 'FOM':
    
        solver_param['solver_mode'] = 'FOM'

    elif input_param['solver_mode'] == 'ROM':

        solver_param['solver_mode']           = 'ROM'
        solver_param['pod_energy']            = float(input_param['pod_energy'])
        solver_param['rom_method']            = input_param['rom_method']     
        solver_param['hyper']                 = eval(input_param['hyper'])         
        solver_param['sampling_method']       = input_param['hyper_method']
        solver_param['fom_results_max_iter']  = int(input_param['fom_results_max_iter'])

        solver_param['fom_results_start']      = int(input_param['fom_results_start'])
        solver_param['fom_results_end']        = int(input_param['fom_results_end'])
        solver_param['fom_results_step']       = int(input_param['fom_results_step'])

    elif  input_param['solver_mode'] == 'Adaptive ROM':

        solver_param['solver_mode']           = 'Adaptive ROM'
        solver_param['adaptive_rom_method']   = input_param['arom_method']
        solver_param['pod_energy']            = float(input_param['pod_energy'])
        solver_param['init_training_win']     = float(input_param['init_training_win'])
        solver_param['unsampled_update_freq'] = int(input_param['unsampled_update_freq'])
        solver_param['rom_method']            = input_param['rom_method']     
        solver_param['hyper']                 = eval(input_param['hyper'])          
        solver_param['sampling_method']       = input_param['hyper_method']

    elif input_param['solver_mode'] == 'Hybrid ROM':

        solver_param['solver_mode']           = 'Hybrid ROM'
        solver_param['adaptive_rom_method']   = input_param['arom_method']
        solver_param['rom_type']              = input_param['rom_type']
        solver_param['pod_energy']            = float(input_param['pod_energy'])
        solver_param['init_training_win']     = float(input_param['init_training_win'])
        solver_param['unsampled_update_freq'] = int(input_param['unsampled_update_freq'])
        solver_param['rom_method']            = input_param['rom_method']     
        solver_param['hyper']                 = eval(input_param['hyper'])          
        solver_param['sampling_method']       = input_param['hyper_method']

        solver_param['fom_results_start']      = int(input_param['fom_results_start'])
        solver_param['fom_results_end']        = int(input_param['fom_results_end'])
        solver_param['fom_results_step']       = int(input_param['fom_results_step'])
        

    ### time discretization ###
    solver_param['dt']          = float(input_param['dt'])
    solver_param['num_step']    = int(input_param['num_steps'])
    solver_param['time_scheme'] = input_param['time_scheme']


    ### space discretization ###
    solver_param['x_initial']   = float(input_param['x_initial'])
    solver_param['x_final']     = float(input_param['x_final'])
    solver_param['cell_number'] = int(input_param['cell_number'])

    ### BC data ###
    solver_param['bc_data'] = np.empty((5,2) , dtype='object')
    
    solver_param['bc_data'][0,0] = input_param['rho_inlet']
    solver_param['bc_data'][1,0] = input_param['vel_inlet']
    solver_param['bc_data'][2,0] = input_param['press_inlet']
    solver_param['bc_data'][3,0] = input_param['temp_inlet']
    solver_param['bc_data'][4,0] = input_param['mass_frac_inlet']


    solver_param['bc_data'][0,1] = input_param['rho_outlet']
    solver_param['bc_data'][1,1] = input_param['vel_outlet']
    solver_param['bc_data'][2,1] = input_param['press_outlet']
    solver_param['bc_data'][3,1] = input_param['temp_outlet']
    solver_param['bc_data'][4,1] = input_param['mass_frac_outlet']

    ### ic data ###

    if 'ic_path' in input_param:

        solver_param['ic_data'] = np.load(input_param['ic_path'])

        solver_param['num_species']   = len(solver_param['ic_data'][:,0])-4

    else:

        # Number of rows
        num_rows = len(eval(input_param['x_interval_ic']))

        # Structuring the data
        solver_param['ic_data'] = []

        for i in range(num_rows):

            row = [str(eval(input_param['x_interval_ic'])[i][0]),
                   str(eval(input_param['x_interval_ic'])[i][1]),
                   str(eval(input_param['rho_ic'])[i]),
                   str(eval(input_param['vel_ic'])[i]),
                   str(eval(input_param['press_ic'])[i]),
                   str(eval(input_param['temp_ic'])[i]),
                   str(eval(input_param['mass_frac_ic'])[i])
                   ]
            
            solver_param['ic_data'].append(row)

        solver_param['num_species']   = len(eval(solver_param['ic_data'][0][6]))

    ### physics ###
    solver_param['gas_model']     =  input_param['gas_model']
    solver_param['gamma']         =  1.4 
    solver_param['flux_scheme']   =  input_param['flux_scheme']
    solver_param['limiter']       =  eval(input_param['limiter'])
    solver_param['limiter_method']=  input_param['limiter_method']
    solver_param['viscous_flag']  =  eval(input_param['viscous'])
    solver_param['injection']     =  eval(input_param['injection'])

    if solver_param['injection']:

        solver_param['injcetion_prim_state']       = np.load(input_param['injection_state_dir'])
        solver_param['injector_face_area']         = float(input_param['injector_face_area'])
        solver_param['non_inj_portion']            = float(input_param['non_injection_portion'])
        solver_param['non_inj_tail_portion']       = float(input_param['non_injection_tail_portion'])

    ### visualization ###
    solver_param['visual']              = eval(input_param['visual'])
    solver_param['variable1']           = input_param['variable1']
    solver_param['variable2']           = input_param['variable2']
    solver_param['variable3']           = input_param['variable3']
    solver_param['variable4']           = input_param['variable4']
    solver_param['vis_update_interval'] = int(input_param['update_interval'])
    solver_param['plot_fom_flag']       = eval(input_param['plot_fom'])
    
    ### Input Directory ###
    solver_param['working_dir']         = args.working_directory


    ### Saving Data ###
    solver_param['save_interval']       = int(input_param['save_interval'])

    ### Saving Data ###
    solver_param['profiling']           = eval(input_param['profiling'])

    
    ### Some Basic Calculations ###

    solver_param['dx']            = (solver_param['x_final'] - solver_param['x_initial']) / solver_param['cell_number']
    solver_param['vol']           = solver_param['dx']
    solver_param['x']             = np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number']) )

    if solver_param['gas_model']  == 'Non-Reacting Air':

        solver_param['num_species']   = 0
        solver_param['num_prim_var']  = 4
        solver_param['num_state_var'] = 3 

    else:

        
        solver_param['num_prim_var']  = 4 + solver_param['num_species']
        solver_param['num_state_var'] = solver_param['num_prim_var'] - 1 # no temp
    
    return solver_param

def initialize_state(solver_param):

    # create state variable
    state = {}

    num_state_var = solver_param['num_state_var']
    num_prim_var  = solver_param['num_prim_var']
    num_species   = solver_param['num_species']

    state['cons_results_save']      = np.zeros(( num_state_var , int(solver_param['cell_number']) ))
    state['prim_results_save']      = np.zeros(( num_prim_var  , int(solver_param['cell_number']) ))
    state['res_save']               = np.zeros(( num_state_var , int(solver_param['cell_number']) ))

    state['Q_cons']                 = np.zeros((num_state_var*int(solver_param['cell_number'])+(num_state_var*4)))
    state['Q_prim']                 = np.zeros((num_prim_var *int(solver_param['cell_number'])+(num_prim_var*4)))

    state['Q_cons_old']             = state['Q_cons']

    if solver_param['gas_model'] != 'Non-Reacting Air':

        mech_file_dir = solver_param['working_dir']+'/chem.yaml'

        # state['gas']       = ct.Solution(mech_file_dir)
        state['gas']         = ct.Solution('h2o2.yaml')
        state['gas'].basis   = 'mass'
        state['gas_array']   = ct.SolutionArray(state['gas'],(1,int(solver_param['cell_number'])+4))

        if solver_param['flux_scheme'] == '2nd Order Roe':
            
            num_cell_with_ghosts = solver_param['cell_number']+4
            num_faces_with_ghosts = num_cell_with_ghosts + 1

            num_subfaces = 2*num_faces_with_ghosts

            state['gas_array_2nd_order'] = ct.SolutionArray(  state['gas'],(1,int(num_subfaces))   )

        # heat release is also plotted part of prim when there is a combustion case
        state['prim_results_save']      = np.zeros(( num_prim_var+1  , int(solver_param['cell_number'])))

    return state

def injection_init(solver_param,state):

    # prepare the number of cells that will be influenced by injection
    state['injection_add_final'] = int(solver_param['non_inj_portion'] * solver_param['cell_number'])
    state['injection_sub_init']  = int(solver_param['non_inj_tail_portion'] * solver_param['cell_number'])
    
    return state

def ic_generator(solver_param,state):

    # already ic profile given

    if np.size(solver_param['ic_data']) == solver_param['num_prim_var'] * int(solver_param['cell_number']):

        full_ic_profile = solver_add_ghost(solver_param['cell_number'],solver_param['num_prim_var'],solver_param['ic_data'].ravel())

        state['Q_prim'] = full_ic_profile

    else:

        num_region = len(solver_param['ic_data'])
        num_cell   = solver_param['cell_number']
        x          =  np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number'])+4 )

        rho = np.zeros(num_cell+4)
        vx  = np.zeros(num_cell+4)
        P   = np.zeros(num_cell+4)
        T   = np.zeros(num_cell+4)
        
        for region in range(0,num_region):

            indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )
        
            rho[indx] = eval(solver_param['ic_data'][region][2])
            vx [indx] = eval(solver_param['ic_data'][region][3])
            P  [indx] = eval(solver_param['ic_data'][region][4])
            T  [indx] = eval(solver_param['ic_data'][region][5])

        state['Q_prim'] = np.vstack((rho,vx,P,T)).ravel()

        state['MW_gas'] = 28.97

        ### In Reacting Flows ###

        if solver_param['gas_model'] != 'Non-Reacting Air':

            MF = np.zeros( (solver_param['num_species'],num_cell+4) )

            for region in range(0,num_region):

                indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )

                MF_list = eval(solver_param['ic_data'][region][6])
                MF_np   = np.array(MF_list)

                for cell in indx[0]:

                    MF[:,cell] = MF_np
                
            state['Q_prim'] = np.vstack((rho,vx,P,T,MF)).ravel()

    return state

def update_ghost_cell(solver_param,state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']
    Q_cons   = state['Q_cons']
    Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

    if solver_param['gas_model'] != 'Non-Reacting Air':

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

    
    Q_prim_solver = results_user2solver_converter(Q_prim_user)
    
    state['Q_prim'] = Q_prim_solver
    state = prim2cons_converter(solver_param,state)

    return state
    
def prim2cons_converter(solver_param, state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        num_cell = solver_param['cell_number']

        Q_prim   = state['Q_prim']

        Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

        vol = solver_param['vol']

        gamma = solver_param['gamma']

        rho = Q_prim_user[0,:]
        vx  = Q_prim_user[1,:] 
        P   = Q_prim_user[2,:]

        mass   = rho * vol
        momx   = rho * vx * vol
        energy = (P/(gamma-1) + 0.5 * rho * (vx**2)) * vol

        state['Q_cons'] = np.vstack((mass,momx,energy)).ravel()

    else:

        num_cell = solver_param['cell_number']

        Q_prim   = state['Q_prim']

        Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

        vol = solver_param['vol']

        rho = Q_prim_user[0,:]
        vx  = Q_prim_user[1,:] 
        P   = Q_prim_user[2,:]
        T   = Q_prim_user[3,:]
        Y   = Q_prim_user[4:,:]

        Y_cantera = find_mass_fraction_full_cantera(Y)
        
        state['gas_array'].TPY = T,P,Y_cantera

        internal_energy = np.squeeze(state['gas_array'].int_energy_mass)

        internal_energy_tot = internal_energy + (0.5 * (vx **2))

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(internal_energy)
        # plt.plot(internal_energy_tot,linestyle='--')

        mass   = rho * vol
        momx   = rho * vx * vol
        energy = (rho * internal_energy_tot) * vol
        mass_species = rho * Y * vol

        state['Q_cons'] = np.vstack((mass,momx,energy,mass_species)).ravel()     

    return state

def cons2prim_converter(solver_param, state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        vol         = solver_param['vol']
        gamma       = solver_param['gamma']

        Q_cons      = state['Q_cons']
        Q_cons_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

        mass        = Q_cons_user[0,:]
        momx        = Q_cons_user[1,:]
        energy      = Q_cons_user[2,:]
        
        rho = mass / vol
        vx  = momx / rho / vol
        P   = (energy/vol - (0.5 * rho * (vx**2))) * (gamma-1)
        T   = P / rho / (ct.gas_constant / state['MW_gas']) 

        state['Q_prim'] = np.vstack((rho,vx,P,T)).ravel()

        # state = update_ghost_cell(solver_param,state)

    else:

        vol         = solver_param['vol']
        gamma       = solver_param['gamma']

        Q_cons      = state['Q_cons']
        Q_cons_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

        mass        = Q_cons_user[0,:]
        momx        = Q_cons_user[1,:]
        energy      = Q_cons_user[2,:]
        mass_species= Q_cons_user[3:,:]
        
        rho = mass / vol
        vx  = momx / rho / vol
        MF  = mass_species / rho / vol

        MF_ct = find_mass_fraction_full_cantera(MF)

        sp_vol = 1/rho

        internal_energy = (energy/vol/rho)-(0.5*vx**2)

        try:

            state['gas_array'].UVY = internal_energy,sp_vol,MF_ct

            T = np.squeeze(state['gas_array'].T)
            P = np.squeeze(state['gas_array'].P)

        except:

            print('Cantera Got Fucked!')

            Q_prim   = state['Q_prim']

            Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],
                                                        solver_param['cell_number'],
                                                        Q_prim)
            
            rho_spare = Q_prim_user[0,:]
            vx_spare  = Q_prim_user[1,:]
            P_spare   = Q_prim_user[2,:]
            T_spare   = Q_prim_user[3,:]
            Y_spare   = Q_prim_user[4:,:]
            Y_spare_ct= find_mass_fraction_full_cantera(Y_spare)

            state['gas_array'].TPY = T_spare, P_spare, Y_spare_ct

            coef_temp  = np.squeeze(state['gas_array'].int_energy_mass)/T_spare
            coef_press = np.squeeze(state['gas_array'].int_energy_mass)/P_spare

            T = np.abs(internal_energy / coef_temp)
            P = np.abs(internal_energy / coef_press)

            # T = np.clip(T,np.min(T_spare),np.max(T_spare))
            # P = np.clip(P,np.min(P_spare),np.max(P_spare))

            # rho = np.clip(rho,np.min(rho_spare),np.max(rho_spare))
            # vx  = np.clip(vx,np.min(vx_spare),np.max(vx_spare))
            # Y   = np.clip(MF,np.min(Y_spare,axis=1)[:,None],np.max(Y_spare,axis=1)[:,None])

            Q_prim = np.vstack((rho,vx,P,T,MF))

            P_max = np.max(P_spare)
            P_min = np.min(P_spare)

            P_max_indx = np.argmax(P_spare)
            P_min_indx = np.argmin(P_spare)

            indx_pass_max = np.where(Q_prim[2,:]>P_max)[0]
            indx_pass_min = np.where(Q_prim[2,:]<P_min)[0]

            Q_prim[:,indx_pass_max] = Q_prim_user[:,P_max_indx].reshape(-1,1)
            Q_prim[:,indx_pass_min] = Q_prim_user[:,P_min_indx].reshape(-1,1)

            rho = Q_prim_user[0,:]
            vx  = Q_prim_user[1,:]
            P   = Q_prim_user[2,:]
            T   = Q_prim_user[3,:]
            Y   = Q_prim_user[4:,:]

            Y_ct= find_mass_fraction_full_cantera(Y)

            state['gas_array'].TPY = T, P, Y_ct

            internal_energy = np.squeeze(state['gas_array'].int_energy_mass)

            internal_energy_tot = internal_energy + (0.5 * (vx **2))

            mass   = rho * vol
            momx   = rho * vx * vol
            energy = (rho * internal_energy_tot) * vol
            mass_species = rho * Y * vol

            state['Q_cons'] = np.vstack((mass,momx,energy,mass_species)).ravel()

        state['Q_prim'] = np.vstack((rho,vx,P,T,MF)).ravel()


    return  state

def slope_limit(df_dx,Q_user,dx,solver_param):

    if solver_param['limiter_method'] == 'minmod':
        
       df_dx = minmod_limiter(df_dx,Q_user,dx)

    elif solver_param['limiter_method'] == 'barth':
        
        df_dx = barth_jespersen(df_dx,Q_user,dx,solver_param)

    return df_dx

def minmod_limiter(df_dx,Q_user,dx):


    df_dx = np.maximum(0., np.minimum(1., ( (Q_user-np.roll(Q_user,1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx
    df_dx = np.maximum(0., np.minimum(1., (-(Q_user-np.roll(Q_user,-1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx


    return df_dx

def barth_jespersen(df_dx,Q_user,dx,solver_param):

    Q_shape  = np.shape(Q_user)

    num_vars = Q_shape[0]
    num_cell = Q_shape[1]

    phi = np.zeros_like(Q_user)

    face_prim_user = np.zeros((num_vars,2*(num_cell)+2))
    
    for indx in range(1,num_cell):

        face_prim_user[:,2*indx]   = Q_user[:,indx-1] + df_dx[:,indx-1] * (dx/2)     # left face 
        face_prim_user[:,2*indx+1] = Q_user[:,indx]   - df_dx[:,indx] * (dx/2)       # right face 

    phi_left1  = 1
    phi_left2  = 1
    phi_left3  = 1
    phi_right1 = 1
    phi_right2 = 1
    phi_right3 = 1
    
    for var in range(num_vars):

        for indx in range(1,num_cell-1):

            max_neighbor = np.max((Q_user[var,indx-1],Q_user[var,indx],Q_user[var,indx+1]))
            min_neighbor = np.min((Q_user[var,indx-1],Q_user[var,indx],Q_user[var,indx+1]))

            left_face_estimate  = face_prim_user[var,2*indx+1]  
            right_face_estimate = face_prim_user[var,2*(indx+1)] 
            
            # left face check 
            if  left_face_estimate - Q_user[var,indx] > 0:

                phi_left1 = np.min(( 1 , (max_neighbor - Q_user[var,indx]) / (left_face_estimate -  Q_user[var,indx]) ))

            elif left_face_estimate - Q_user[var,indx] < 0:

                phi_left2 = np.min(( 1 , (min_neighbor - Q_user[var,indx]) / (left_face_estimate -  Q_user[var,indx]) ))

            else:
                
                phi_left3 = 1


            # right face check 
            if  right_face_estimate - Q_user[var,indx] > 0:

                phi_right1 = np.min(( 1 , (max_neighbor - Q_user[var,indx]) / (right_face_estimate -  Q_user[var,indx]) ))

            elif right_face_estimate - Q_user[var,indx] < 0:

                phi_right2 = np.min(( 1 , (min_neighbor - Q_user[var,indx]) / (right_face_estimate -  Q_user[var,indx]) ))

            else:
                
                phi_right3 = 1

            phi[var,indx]=np.min((phi_left1,phi_left2,phi_left3,phi_right1,phi_right2,phi_right3))


    df_dx = df_dx * phi

    return df_dx

def results_recorder(solver_param, rom_param, state):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    if solver_param['solver_mode'] == 'FOM':
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, f"{save_title}_cons.npy"), state['cons_results_save'])
        np.save(os.path.join(dir_results, f"{save_title}_prim.npy"), state['prim_results_save'])
        np.save(os.path.join(dir_results, f"{save_title}_res.npy") , state['res_save'])

    elif solver_param['solver_mode'] == 'ROM' and iter == 0:
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_cons.npy")           , state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_prim.npy")           , state['prim_results_save'])
        np.save(os.path.join(dir_results, 'res'           , f"{save_title}_res.npy")            , state['res_save']         )
        np.save(os.path.join(dir_results, 'q_r'           , f"{save_title}_q_r.npy")            , rom_param['q_red0']       )
        np.save(os.path.join(dir_results, 'basis'         , f"{save_title}_basis.npy")          , rom_param['basis']        )
        np.save(os.path.join(dir_results, 'q_ref'         , f"{save_title}_q_ref.npy")          , rom_param['q_ref']        )
        np.save(os.path.join(dir_results, 'norm'          , f"{save_title}_norm.npy")           , rom_param['normalizor']   )
        np.save(os.path.join(dir_results, 'denorm'        , f"{save_title}_denorm.npy")         , rom_param['denormalizor'] )
        np.save(os.path.join(dir_results, 'samples_user'  , f"{save_title}_samples_user.npy")   , rom_param['S_indx_user']  )
        np.save(os.path.join(dir_results, 'samples_solver', f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])

    elif solver_param['solver_mode'] == 'ROM':
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_cons.npy"), state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_prim.npy"), state['prim_results_save'])
        np.save(os.path.join(dir_results, 'res'           , f"{save_title}_res.npy")            , state['res_save']         )
        np.save(os.path.join(dir_results, 'res'      , f"{save_title}_res.npy") , state['res_save']         )
        np.save(os.path.join(dir_results, 'q_r'      , f"{save_title}_q_r.npy") , rom_param['q_red0'])

    elif (solver_param['solver_mode'] == 'Adaptive ROM' and iter < int(solver_param['init_training_win'])):
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_cons.npy"), state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_prim.npy"), state['prim_results_save'])
        np.save(os.path.join(dir_results, 'res'      , f"{save_title}_res.npy") , state['res_save'])

    elif (solver_param['solver_mode'] == 'Adaptive ROM' and iter == int(solver_param['init_training_win'])):
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim'      , f"{save_title}_cons.npy")           , state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim'      , f"{save_title}_prim.npy")           , state['prim_results_save'])
        np.save(os.path.join(dir_results, 'res'            , f"{save_title}_res.npy")            , state['res_save']         )        
        np.save(os.path.join(dir_results, 'basis'          , f"{save_title}_basis.npy")          , rom_param['basis']        )
        np.save(os.path.join(dir_results, 'q_ref'          , f"{save_title}_q_ref.npy")          , rom_param['q_ref']        )
        np.save(os.path.join(dir_results, 'norm'           , f"{save_title}_norm.npy")           , rom_param['normalizor']   )
        np.save(os.path.join(dir_results, 'denorm'         , f"{save_title}_denorm.npy")         , rom_param['denormalizor'] )
        np.save(os.path.join(dir_results, 'samples_user'   , f"{save_title}_samples_user.npy")   , rom_param['S_indx_user']  )
        np.save(os.path.join(dir_results, 'samples_solver' , f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])

        # q_r needed for init training windows
        for indx in range(int(solver_param['init_training_win']) + 1):
            save_title = 'iteration' + str(indx)
            np.save(os.path.join(dir_results, 'q_r', f"{save_title}_q_r.npy"), rom_param['q_red0'])

    elif (solver_param['solver_mode'] == 'Adaptive ROM' and iter > int(solver_param['init_training_win'])):
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_cons.npy")          , state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_prim.npy")          , state['prim_results_save'])
        np.save(os.path.join(dir_results, 'res'           , f"{save_title}_res.npy")           , state['res_save']         )
        np.save(os.path.join(dir_results, 'q_r'           , f"{save_title}_q_r.npy")           , rom_param['q_red0']       )
        np.save(os.path.join(dir_results, 'basis'         , f"{save_title}_basis.npy")         , rom_param['basis']        )
        np.save(os.path.join(dir_results, 'samples_user'  , f"{save_title}_samples_user.npy")  , rom_param['S_indx_user']  )
        np.save(os.path.join(dir_results, 'samples_solver', f"{save_title}_samples_solver.npy"), rom_param['S_indx_solver'])  

    elif (solver_param['solver_mode'] == 'Hybrid ROM' and iter > int(solver_param['init_training_win'])):
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_cons.npy")          , state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_prim.npy")          , state['prim_results_save'])
        np.save(os.path.join(dir_results, 'res'           , f"{save_title}_res.npy")           , state['res_save']         )
        np.save(os.path.join(dir_results, 'q_r'           , f"{save_title}_q_r.npy")           , rom_param['q_red0']       )
        np.save(os.path.join(dir_results, 'basis'         , f"{save_title}_basis.npy")         , rom_param['basis']        )
        np.save(os.path.join(dir_results, 'samples_user'  , f"{save_title}_samples_user.npy")  , rom_param['S_indx_user']  )
        np.save(os.path.join(dir_results, 'samples_solver', f"{save_title}_samples_solver.npy"), rom_param['S_indx_solver']) 

def first_order_roe_inviscid_flux_calculator_for(solver_param,rom_param,state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)
    
        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]

        
        gamma  = solver_param['gamma']
        vol    = solver_param['vol']
        num_state_var = solver_param['num_state_var']

        # breakpoint()

        # c_max = np.max(np.abs(np.sqrt(gamma*press/rho)))

        # cfl = c_max*solver_param['dt']/vol

        # dt = 0.08*vol/c_max

        S_indx_user = rom_param['S_indx_user']

        en               = press  / (gamma-1) + 0.5 * rho  * (vx**2)
        # number of cell
        cell_num = len(rho)

        # total enthalpy
        htot = gamma/(gamma-1)*press/rho+0.5*vx**2

        flux = np.zeros((num_state_var,cell_num+1))
        diffusion = np.zeros((num_state_var,cell_num+1))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))


        else : 
            
            range_flux = range(0,cell_num-1)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
            hmoy=(R*htot[j+1]+htot[j])/(R+1);               # {hat H}_{j+1/2}
            amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy))  # {hat a}_{j+1/2}
            
            # Auxiliary variables used to compute P_{j+1/2}^{-1}
            alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
            alph2=(gamma-1)/(amoy*amoy)

            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                            [1-alph1,                alph2*umoy,                -alph2 ],
                            [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]])
                    
            # Compute matrix P_{j+1/2}
            P    = np.array([[ 1,              1,              1              ],
                            [umoy-amoy,        umoy,           umoy+amoy      ],
                            [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]])
            
            # Compute matrix Lambda_{j+1/2}
            lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                            [0,                 abs(umoy),      0                 ],
                            [0,                 0,              abs(umoy+amoy)    ]])
                        
            # Compute Roe matrix |A_{j+1/2}|
            A = P @ lamb @ Pinv

            diffusion[:,j+1] = 0.5 * A @ (Q_cons_user[:,j+1]-Q_cons_user[:,j]) / vol

            flux[0,j+1] = 0.5*(rho[j]*vx[j]               + rho[j+1]*vx[j+1])                         - diffusion[0,j+1] 
            flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]**2+press[j+1])           - diffusion[1,j+1] 
            flux[2,j+1] = 0.5*(vx[j]*(en[j]+press[j])     + vx[j+1]*(en[j+1]+press[j+1]))             - diffusion[2,j+1] 

    ### If It is a Multi-Species Case ###

    else :
        
        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)

        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]
        temp   = Q_prim_user[3,:]


        Y      = Q_prim_user[4:,:]
        Y_ct   = find_mass_fraction_full_cantera(Y)


        vol    = solver_param['vol']
        num_state_var = solver_param['num_state_var']
        num_species   = solver_param['num_species']



        S_indx_user = rom_param['S_indx_user']

        # total energy (rho * e_t)
        en               = Q_cons_user[2,:] / vol

        state['gas_array'].TPY = temp,press,Y_ct

        c       = np.squeeze(state['gas_array'].sound_speed)
        int_en  = np.squeeze(state['gas_array'].int_energy_mass)
        h       = np.squeeze(state['gas_array'].enthalpy_mass)

        # breakpoint()

        # c_max = np.max(c)

        # cfl = c_max*1e-8/vol

        # dt = 0.08*vol/c_max

        # total enthalpy
        htot = h + (0.5*(vx**2))

        # number of cell
        cell_num = len(rho)

        flux = np.zeros((num_state_var,cell_num+1))
        diss_matrix = np.zeros((num_state_var,num_state_var))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))

        else : 
            
            range_flux = range(0,cell_num-1)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
            Hmoy=(R*htot[j+1]+htot[j])/(R+1)                # {hat H}_{j+1/2}
            emoy=(R*int_en[j+1]+int_en[j])/(R+1)              
            hmoy=Hmoy - (0.5*umoy*umoy)
            Ymoy=(R*Y[:,j+1]+Y[:,j])/(R+1)                  

            Ymoy_ct = find_mass_fraction_full_cantera(Ymoy.reshape(-1,1))

            state['gas'].UVY = emoy,1/rmoy,Ymoy_ct

            Pmoy         = state['gas'].P
            Tmoy         = state['gas'].T
            cpmoy        = state['gas'].cp
            cmoy         = state['gas'].sound_speed
            meanMWmoy    = state['gas'].mean_molecular_weight
            MWmoy        = state['gas'].molecular_weights
            partial_hmoy = state['gas'].partial_molar_enthalpies / MWmoy
            
            d_rho_d_press = rmoy / Pmoy
            d_rho_d_temp = -rmoy/Tmoy

            d_rho_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_rho_d_mass_frac[sp] = rmoy*meanMWmoy*(1/MWmoy[-1] - 1/MWmoy[sp])

            d_enth_d_press = 0
            d_enth_d_temp = cpmoy

            d_enth_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_enth_d_mass_frac[sp] = partial_hmoy[sp] - partial_hmoy[-1]

            # Gamma terms for energy equation
            g_press     = rmoy * d_enth_d_press + d_rho_d_press * Hmoy - 1.0
            g_temp      = rmoy * d_enth_d_temp + d_rho_d_temp * Hmoy
            g_mass_frac = rmoy * d_enth_d_mass_frac + Hmoy * d_rho_d_mass_frac

            # Characteristic speeds
            lambda1 = umoy + cmoy
            lambda2 = umoy - cmoy
            lambda1_abs = np.absolute(lambda1)
            lambda2_abs = np.absolute(lambda2)

            r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
            alpha = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
            beta  = np.power(cmoy, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
            phi   = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

            eta = (1.0 - rmoy * d_enth_d_press) / d_enth_d_temp
            psi = eta * d_rho_d_temp + rmoy * d_rho_d_press

            vel_abs = np.absolute(umoy)

            beta_star = beta * psi
            beta_e = beta * (rmoy * g_press + g_temp * eta)
            phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rmoy
            phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rmoy
            m = rmoy * alpha
            e = rmoy * umoy * alpha

            delta_p = press[j]-press[j+1]
            delta_u = vx[j]-vx[j+1]
            delta_T = temp[j]-temp[j+1]
            delta_Y = (Y[:,j]-Y[:,j+1]).reshape(-1,1)


            del_q_prim = np.vstack((delta_p,delta_u,delta_T,delta_Y))

            diss_matrix[ 0 , 0 ] = phi_star
            diss_matrix[ 0 , 1 ] = beta_star
            diss_matrix[ 0 , 2 ] = vel_abs * d_rho_d_temp
            diss_matrix[ 0 , 3:] = vel_abs * d_rho_d_mass_frac

            diss_matrix[ 1 , 0 ] = umoy * phi_star + r_roe
            diss_matrix[ 1 , 1 ] = umoy * beta_star + m
            diss_matrix[ 1 , 2 ] = umoy * vel_abs * d_rho_d_temp
            diss_matrix[ 1 , 3:] = (umoy * vel_abs) * d_rho_d_mass_frac

            diss_matrix[ 2 , 0 ] = phi_e + r_roe * umoy
            diss_matrix[ 2 , 1 ] = beta_e + e
            diss_matrix[ 2 , 2 ] = g_temp * vel_abs
            diss_matrix[ 2 , 3:] = g_mass_frac * vel_abs

            diss_matrix[3:, 0] = Ymoy * phi_star
            diss_matrix[3:, 1] = Ymoy * beta_star
            diss_matrix[3:, 2] = Ymoy * (vel_abs * d_rho_d_temp)

            for mf_idx_out in range(3, num_state_var):

                for mf_idx_in in range(3, num_state_var):

                    if mf_idx_out == mf_idx_in:
                        diss_matrix[mf_idx_out, mf_idx_in] = vel_abs * (
                            rmoy + Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )
                    else:
                        diss_matrix[mf_idx_out, mf_idx_in] = (
                            vel_abs * Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )

            dissipation = diss_matrix @ del_q_prim
        
            flux[0,j+1] = 0.5*(rho[j]*vx[j]               + rho[j+1]*vx[j+1])                         + 0.5 * dissipation[0 ,0]
            flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]**2+press[j+1])           + 0.5 * dissipation[1 ,0]
            flux[2,j+1] = 0.5*(vx[j]*(en[j]+press[j])     + vx[j+1]*(en[j+1]+press[j+1]))             + 0.5 * dissipation[2 ,0]
            flux[3:,j+1]= 0.5*(rho[j]*vx[j]*Y[:,j]        + rho[j+1]*vx[j+1]*Y[:,j+1] )               + 0.5 * dissipation[3:,0]
            

    state['flux_cons'] = flux

    return state

def first_order_roe_inviscid_flux_calculator(solver_param,rom_param,state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)
    
        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]

        
        gamma         = solver_param['gamma']
        vol           = solver_param['vol']
        num_state_var = solver_param['num_state_var']

        S_indx_user   = rom_param['S_indx_user']

        en               = press  / (gamma-1) + 0.5 * rho  * (vx**2)

        cell_num = len(rho)

        # total enthalpy
        htot = gamma/(gamma-1)*press/rho+0.5*vx**2

        # flux = np.zeros((num_state_var,cell_num+1))
        # diffusion = np.zeros((num_state_var,cell_num+1))

        if solver_param['hyper'] == True:

            S_indx_user      = S_indx_user + 2
            range_flux_right = np.sort(np.concatenate((S_indx_user,S_indx_user+1)))
            range_flux_left  = range_flux_right - 1

            rho_left  = rho[range_flux_left]
            rho_right = rho[range_flux_right]

            vx_left  = vx[range_flux_left]
            vx_right = vx[range_flux_right]

            htot_left  = htot[range_flux_left]
            htot_right = htot[range_flux_right]

            press_left  = press[range_flux_left]
            press_right = press[range_flux_right]

            en_left    = en[range_flux_left]
            en_right   = en[range_flux_right]

            left_state = Q_cons_user[:,range_flux_left]
            right_state= Q_cons_user[:,range_flux_right]

            Pinv =  np.zeros((2*len(S_indx_user),3,3))
            P    =  np.zeros((2*len(S_indx_user),3,3))
            lamb =  np.zeros((2*len(S_indx_user),3,3))

        else : 
            
            rho_left  = rho[:-1]
            rho_right = rho[1:]

            vx_left  = vx[:-1]
            vx_right = vx[1:]

            htot_left  = htot[:-1]
            htot_right = htot[1:]

            press_left  = press[:-1]
            press_right = press[1:]

            en_left    = en[:-1]
            en_right   = en[1:]

            left_state = Q_cons_user[:,:-1]
            right_state= Q_cons_user[:,1:]

            Pinv =  np.zeros((cell_num-1,3,3))
            P    =  np.zeros((cell_num-1,3,3))
            lamb =  np.zeros((cell_num-1,3,3))

        R = np.sqrt(rho_right/rho_left)

        rmoy=R*rho_left                                  # {hat rho}_{j+1/2}
        umoy=(R*vx_right+vx_left)/(R+1)                    # {hat U}_{j+1/2}
        hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
        amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy))   # {hat a}_{j+1/2}
    
        alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
        alph2=(gamma-1)/(amoy*amoy)

        Pinv[:,0,0] = 0.5*(alph1+umoy/amoy)
        Pinv[:,0,1] = -0.5*(alph2*umoy+1/amoy)
        Pinv[:,0,2] = alph2/2

        Pinv[:,1,0] = 1-alph1
        Pinv[:,1,1] = alph2*umoy
        Pinv[:,1,2] = -alph2

        Pinv[:,2,0] = 0.5*(alph1-umoy/amoy)
        Pinv[:,2,1] = -0.5*(alph2*umoy-1/amoy)
        Pinv[:,2,2] = alph2/2
        
        P[:,0,0] = 1
        P[:,0,1] = 1
        P[:,0,2] = 1

        P[:,1,0] = umoy-amoy
        P[:,1,1] = umoy
        P[:,1,2] = umoy+amoy

        P[:,2,0] = hmoy-amoy*umoy
        P[:,2,1] = 0.5*umoy*umoy
        P[:,2,2] = hmoy+amoy*umoy

        lamb[:,0,0] = abs(umoy-amoy)
        lamb[:,0,1] = 0
        lamb[:,0,2] = 0

        lamb[:,1,0] = 0
        lamb[:,1,1] = abs(umoy)
        lamb[:,1,2] = 0

        lamb[:,2,0] = 0
        lamb[:,2,1] = 0
        lamb[:,2,2] = abs(umoy+amoy)

        A = np.matmul(np.matmul(P,lamb),Pinv)

        dq = ((right_state-left_state).T)

        dq = dq[:,:,np.newaxis]

        diffusion = 0.5 * np.matmul(A,dq) / vol

        flux = np.zeros((num_state_var,cell_num+1))
        
        if solver_param['hyper'] == True:

            flux[0,range_flux_right] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                   - np.squeeze(diffusion[:,0,:])
            flux[1,range_flux_right] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)    - np.squeeze(diffusion[:,1,:])
            flux[2,range_flux_right] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))      - np.squeeze(diffusion[:,2,:])

        else:

            flux[0,1:-1] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                   - np.squeeze(diffusion[:,0,:])
            flux[1,1:-1] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)    - np.squeeze(diffusion[:,1,:])
            flux[2,1:-1] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))      - np.squeeze(diffusion[:,2,:])

    ### If It is a Multi-Species Case ###

    else :
        
        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)

        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]
        temp   = Q_prim_user[3,:]


        Y      = Q_prim_user[4:,:]
        Y_ct   = find_mass_fraction_full_cantera(Y)


        vol           = solver_param['vol']
        num_state_var = solver_param['num_state_var']
        num_species   = solver_param['num_species']

        S_indx_user = rom_param['S_indx_user']

        # total energy (rho * e_t)
        en               = Q_cons_user[2,:] / vol

        state['gas_array'].TPY = temp,press,Y_ct

        c       = np.squeeze(state['gas_array'].sound_speed)
        int_en  = np.squeeze(state['gas_array'].int_energy_mass)
        h       = np.squeeze(state['gas_array'].enthalpy_mass)

        # total enthalpy
        htot = h + (0.5*(vx**2))

        # number of cell
        cell_num = len(rho)

        flux = np.zeros((num_state_var,cell_num+1))
        

        if solver_param['hyper'] == True:

            S_indx_user      = S_indx_user + 2
            range_flux_right = np.sort(np.concatenate((S_indx_user,S_indx_user+1)))
            range_flux_left  = range_flux_right - 1

            rho_left  = rho[range_flux_left]
            rho_right = rho[range_flux_right]

            vx_left  = vx[range_flux_left]
            vx_right = vx[range_flux_right]

            htot_left  = htot[range_flux_left]
            htot_right = htot[range_flux_right]

            press_left  = press[range_flux_left]
            press_right = press[range_flux_right]

            temp_left  = temp[range_flux_left]
            temp_right = temp[range_flux_right]

            en_left    = en[range_flux_left]
            en_right   = en[range_flux_right]

            int_en_left    = int_en[range_flux_left]
            int_en_right   = int_en[range_flux_right]

            Y_left         = Y[:,range_flux_left]
            Y_right        = Y[:,range_flux_right]

            left_state = Q_cons_user[:,range_flux_left]
            right_state= Q_cons_user[:,range_flux_right]

            del_q_prim = np.zeros((2*len(S_indx_user),num_state_var,1))

            Pinv =  np.zeros((2*len(S_indx_user),3,3))
            P    =  np.zeros((2*len(S_indx_user),3,3))
            lamb =  np.zeros((2*len(S_indx_user),3,3))

            diss_matrix = np.zeros((2*len(S_indx_user),num_state_var,num_state_var))

            gas_array = ct.SolutionArray(state['gas'],2*len(S_indx_user))

        else : 
            
            rho_left  = rho[:-1]
            rho_right = rho[1:]

            vx_left  = vx[:-1]
            vx_right = vx[1:]

            htot_left  = htot[:-1]
            htot_right = htot[1:]

            press_left  = press[:-1]
            press_right = press[1:]

            temp_left  = temp[:-1]
            temp_right = temp[1:]

            en_left    = en[:-1]
            en_right   = en[1:]

            int_en_left  = int_en[:-1]
            int_en_right = int_en[1:]

            Y_left         = Y[:,:-1]
            Y_right        = Y[:,1:]

            left_state = Q_cons_user[:,:-1]
            right_state= Q_cons_user[:,1:]

            del_q_prim = np.zeros((cell_num-1,num_state_var,1))

            Pinv =  np.zeros((cell_num-1,3,3))
            P    =  np.zeros((cell_num-1,3,3))
            lamb =  np.zeros((cell_num-1,3,3))

            diss_matrix = np.zeros((cell_num-1,num_state_var,num_state_var))

            gas_array = ct.SolutionArray(state['gas'],cell_num-1)
        
        R   = np.sqrt(rho_right/rho_left)

        rmoy=R*rho_left                                   # {hat rho}_{j+1/2}
        umoy=(R*vx_right+vx_left)/(R+1)                   # {hat U}_{j+1/2}
        Hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
        emoy=(R*int_en_right+int_en_left)/(R+1)              
        hmoy=Hmoy - (0.5*umoy*umoy)
        Ymoy=(R*Y_right+Y_left)/(R+1)                  

        Ymoy_ct = find_mass_fraction_full_cantera(Ymoy)

        gas_array.UVY = emoy,1/rmoy,Ymoy_ct

        Pmoy         = gas_array.P
        Tmoy         = gas_array.T
        cpmoy        = gas_array.cp
        cmoy         = gas_array.sound_speed
        meanMWmoy    = gas_array.mean_molecular_weight
        MWmoy        = gas_array.molecular_weights
        partial_hmoy = gas_array.partial_molar_enthalpies / MWmoy
            
        d_rho_d_press = rmoy / Pmoy
        d_rho_d_temp = -rmoy/Tmoy

        # d_rho_d_mass_frac = np.zeros_like(Ymoy)

        #     for sp in range(len(Ymoy)):

        #         d_rho_d_mass_frac[sp] = rmoy*meanMWmoy*(1/MWmoy[-1] - 1/MWmoy[sp])
        d_rho_d_mass_frac_const_term =  (1/MWmoy[-1] - 1/MWmoy[:-1])
        d_rho_d_mass_frac = rmoy * meanMWmoy * d_rho_d_mass_frac_const_term[:,np.newaxis]

        d_enth_d_press = 0
        d_enth_d_temp  = cpmoy

        # d_enth_d_mass_frac = np.zeros_like(Ymoy)

        # for sp in range(len(Ymoy)):
        
        d_enth_d_mass_frac = partial_hmoy[:,:-1] - partial_hmoy[:,-1,np.newaxis]

        #     # Gamma terms for energy equation
        g_press     = rmoy * d_enth_d_press + d_rho_d_press * Hmoy - 1.0
        g_temp      = rmoy * d_enth_d_temp + d_rho_d_temp * Hmoy
        g_mass_frac = rmoy[:,np.newaxis] * d_enth_d_mass_frac + Hmoy[:,np.newaxis] * d_rho_d_mass_frac.T

        #     # Characteristic speeds
        lambda1 = umoy + cmoy
        lambda2 = umoy - cmoy
        lambda1_abs = np.absolute(lambda1)
        lambda2_abs = np.absolute(lambda2)

        r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
        alpha = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
        beta  = np.power(cmoy, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
        phi   = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

        eta = (1.0 - rmoy * d_enth_d_press) / d_enth_d_temp
        psi = eta * d_rho_d_temp + rmoy * d_rho_d_press

        vel_abs = np.absolute(umoy)

        beta_star = beta * psi
        beta_e = beta * (rmoy * g_press + g_temp * eta)
        phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rmoy
        phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rmoy
        m = rmoy * alpha
        e = rmoy * umoy * alpha

        delta_p = press_left-press_right
        delta_u = vx_left-vx_right
        delta_T = temp_left-temp_right
        delta_Y = (Y_left-Y_right)

        # del_q_prim = np.zeros((cell_num-1,num_state_var,1))

        del_q_prim[:,0,:]  = delta_p.reshape(-1,1)
        del_q_prim[:,1,:]  = delta_u.reshape(-1,1)
        del_q_prim[:,2,:]  = delta_T.reshape(-1,1)
        del_q_prim[:,3:,:] = delta_Y.T[:,:,np.newaxis]


        diss_matrix[: , 0 , 0 ] = phi_star
        diss_matrix[: , 0 , 1 ] = beta_star
        diss_matrix[: , 0 , 2 ] = vel_abs * d_rho_d_temp
        diss_matrix[: , 0 , 3:] = vel_abs[:,np.newaxis] * d_rho_d_mass_frac.T

        diss_matrix[: , 1 , 0 ] = umoy * phi_star + r_roe
        diss_matrix[: , 1 , 1 ] = umoy * beta_star + m
        diss_matrix[: , 1 , 2 ] = umoy * vel_abs * d_rho_d_temp
        diss_matrix[: , 1 , 3:] = (umoy * vel_abs)[:,np.newaxis] * d_rho_d_mass_frac.T

        diss_matrix[: , 2 , 0 ] = phi_e + r_roe * umoy
        diss_matrix[: , 2 , 1 ] = beta_e + e
        diss_matrix[: , 2 , 2 ] = g_temp * vel_abs
        diss_matrix[: , 2 , 3:] = g_mass_frac * vel_abs[:,np.newaxis]

        diss_matrix[: , 3:, 0] = Ymoy.T * phi_star[:,np.newaxis]
        diss_matrix[: , 3:, 1] = Ymoy.T * beta_star[:,np.newaxis]
        diss_matrix[: , 3:, 2] = Ymoy.T * (vel_abs * d_rho_d_temp)[:,np.newaxis]


        idx_out, idx_in = np.meshgrid(np.arange(num_state_var-3), 
                                    np.arange(num_state_var-3), indexing='ij')
        
        diag_mask = idx_out == idx_in

        off_diag = vel_abs[:,np.newaxis] * Ymoy.T * d_rho_d_mass_frac.T
        off_diag = np.repeat(off_diag[:,:,np.newaxis],np.shape(off_diag)[1],axis=2)

        diag = vel_abs[:,np.newaxis] * (rmoy[:,np.newaxis] + Ymoy.T * d_rho_d_mass_frac.T)

        diss_matrix[:,3:,3:] = off_diag
        diss_matrix[:,3:,3:][:,diag_mask] = diag

        dissipation = np.matmul(diss_matrix , del_q_prim)

        if solver_param['hyper'] == True:

            flux[0 ,range_flux_right] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                 + 0.5 * dissipation[:,0 ,0]
            flux[1 ,range_flux_right] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)  + 0.5 * dissipation[:,1 ,0]
            flux[2 ,range_flux_right] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))    + 0.5 * dissipation[:,2 ,0]
            flux[3:,range_flux_right] = 0.5*(rho_left*vx_left*Y_left          + rho_right*vx_right*Y_right )        + 0.5 * dissipation[:,3:,0].T


        else:

            flux[0,1:-1] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                 + 0.5 * dissipation[:,0 ,0]
            flux[1,1:-1] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)  + 0.5 * dissipation[:,1 ,0]
            flux[2,1:-1] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))    + 0.5 * dissipation[:,2 ,0]
            flux[3:,1:-1]= 0.5*(rho_left*vx_left*Y_left          + rho_right*vx_right*Y_right )        + 0.5 * dissipation[:,3:,0].T
            

    state['flux_cons'] = flux

    return state

def first_order_roe_viscous_flux_calculator_for(solver_param,rom_param,state):

    # state            = cons2prim_converter(solver_param,state)
    # state            = update_ghost_cell(solver_param,state)

    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]

    Y      = Q_prim_user[4:,:]
    Y_full = find_mass_fraction_full(Y)
    Y_ct   = find_mass_fraction_full_cantera(Y)


    vol           = solver_param['vol']
    dx            = vol
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']

    S_indx_user = rom_param['S_indx_user']

    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    state['gas_array'].TPY = temp,press,Y_ct

    c       = np.squeeze(state['gas_array'].sound_speed)
    int_en  = np.squeeze(state['gas_array'].int_energy_mass)
    h       = np.squeeze(state['gas_array'].enthalpy_mass)

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))

    if solver_param['hyper'] == True:

        range_flux = S_indx_user + 2 
        range_flux_neighbor_left  = range_flux - 1
        range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
        range_flux = np.sort(np.unique(range_flux))


    else : 
        
        range_flux = range(0,cell_num-1)

    for j in range_flux:
    
        # Compute Roe averages
        R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
        rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
        umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
        Hmoy=(R*htot[j+1]+htot[j])/(R+1);               # {hat H}_{j+1/2}
        hmoy=(R*h[j+1]+h[j])/(R+1);                     # {hat h}_{j+1/2}
        emoy=(R*int_en[j+1]+int_en[j])/(R+1);           # {hat e}_{j+1/2}
        Ymoy=(R*Y[:,j+1]+Y[:,j])/(R+1);                 # {hat Y}_{j+1/2}          
        Pmoy = (hmoy - emoy)*rmoy                       # {hat P}_{j+1/2}

        Ymoy_full = np.squeeze( find_mass_fraction_full(Ymoy.reshape(-1,1)) )

        state['gas'].UVY            = emoy,1/rmoy,Ymoy_full

        dyn_vsc_mix                 = state['gas'].viscosity
        mass_diff_mix               = state['gas'].mix_diff_coeffs_mass
        therm_cond_mix              = state['gas'].thermal_conductivity

        MW                          = state['gas'].molecular_weights
        MW_mix                      = state['gas'].mean_molecular_weight
        enthalpies                  = state['gas'].partial_molar_enthalpies / MW


        du_dx = (vx[j+1] - vx[j]) /dx
        dT_dx = (temp[j+1] - temp[j]) /dx
        dY_dx = (Y_full[:,j+1] - Y_full[:,j]) / dx

        corr_vel  = np.sum(mass_diff_mix*dY_dx)

        diff_vel  = -mass_diff_mix*dY_dx/Ymoy_full + corr_vel

        tau       = 4/3 * dyn_vsc_mix * du_dx

        q         = -therm_cond_mix * dT_dx + rmoy*np.sum(enthalpies*diff_vel*Ymoy_full)

        flux[0,j+1] = 0 
        flux[1,j+1] = tau
        flux[2,j+1] = (umoy * tau) - q
        flux[3:,j+1]= (rmoy*mass_diff_mix[:-1]*dY_dx[:-1])-(rmoy*corr_vel*Ymoy)

    state['flux_visc_cons'] = flux

    return state

def first_order_roe_viscous_flux_calculator(solver_param,rom_param,state):

    # state            = cons2prim_converter(solver_param,state)
    # state            = update_ghost_cell(solver_param,state)

    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]

    Y      = Q_prim_user[4:,:]
    Y_full = find_mass_fraction_full(Y)
    Y_ct   = find_mass_fraction_full_cantera(Y)


    vol           = solver_param['vol']
    dx            = vol
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']

    S_indx_user = rom_param['S_indx_user']

    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    state['gas_array'].TPY = temp,press,Y_ct

    c       = np.squeeze(state['gas_array'].sound_speed)
    int_en  = np.squeeze(state['gas_array'].int_energy_mass)
    h       = np.squeeze(state['gas_array'].enthalpy_mass)

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))

    if solver_param['hyper'] == True:

        S_indx_user      = S_indx_user + 2
        range_flux_right = np.sort(np.concatenate((S_indx_user,S_indx_user+1)))
        range_flux_left  = range_flux_right - 1

        rho_left  = rho[range_flux_left]
        rho_right = rho[range_flux_right]

        vx_left  = vx[range_flux_left]
        vx_right = vx[range_flux_right]

        htot_left  = htot[range_flux_left]
        htot_right = htot[range_flux_right]

        press_left  = press[range_flux_left]
        press_right = press[range_flux_right]

        temp_left  = temp[range_flux_left]
        temp_right = temp[range_flux_right]

        en_left    = en[range_flux_left]
        en_right   = en[range_flux_right]

        int_en_left    = int_en[range_flux_left]
        int_en_right   = int_en[range_flux_right]

        Y_left         = Y[:,range_flux_left]
        Y_right        = Y[:,range_flux_right]

        Y_full_left         = Y_full[:,range_flux_left]
        Y_full_right        = Y_full[:,range_flux_right]

        left_state = Q_cons_user[:,range_flux_left]
        right_state= Q_cons_user[:,range_flux_right]  
   
        gas_array = ct.SolutionArray(state['gas'],2*len(S_indx_user))   

    else : 
        
        rho_left  = rho[:-1]
        rho_right = rho[1:]

        vx_left  = vx[:-1]
        vx_right = vx[1:]

        htot_left  = htot[:-1]
        htot_right = htot[1:]

        press_left  = press[:-1]
        press_right = press[1:]

        temp_left  = temp[:-1]
        temp_right = temp[1:]

        en_left    = en[:-1]
        en_right   = en[1:]

        int_en_left  = int_en[:-1]
        int_en_right = int_en[1:]

        Y_left         = Y[:,:-1]
        Y_right        = Y[:,1:]

        Y_full_left         = Y_full[:,:-1]
        Y_full_right        = Y_full[:,1:]

        left_state = Q_cons_user[:,:-1]
        right_state= Q_cons_user[:,1:]

        gas_array = ct.SolutionArray(state['gas'],cell_num-1)


    R = np.sqrt(rho_right/rho_left)

    rmoy=R*rho_left                                   # {hat rho}_{j+1/2}
    umoy=(R*vx_right+vx_left)/(R+1)                   # {hat U}_{j+1/2}
    Hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
    emoy=(R*int_en_right+int_en_left)/(R+1)              
    hmoy=Hmoy - (0.5*umoy*umoy)
    Ymoy=(R*Y_right+Y_left)/(R+1)                  

    Ymoy_ct = find_mass_fraction_full_cantera(Ymoy)

    gas_array.UVY = emoy,1/rmoy,Ymoy_ct

    dyn_vsc_mix                 = gas_array.viscosity
    mass_diff_mix               = gas_array.mix_diff_coeffs_mass
    therm_cond_mix              = gas_array.thermal_conductivity

    MW                          = gas_array.molecular_weights
    MW_mix                      = gas_array.mean_molecular_weight
    enthalpies                  = gas_array.partial_molar_enthalpies / MW


    du_dx = (vx_right - vx_left) /dx
    dT_dx = (temp_right - temp_left) /dx
    dY_dx = (Y_full_right - Y_full_left) / dx

    corr_vel  = np.sum(mass_diff_mix*dY_dx.T,axis=1)

    diff_vel  = -mass_diff_mix*dY_dx.T/Ymoy_ct[0,:,:] + corr_vel[:,np.newaxis]

    tau       = 4/3 * dyn_vsc_mix * du_dx

    q         = -therm_cond_mix * dT_dx + rmoy*np.sum(enthalpies*diff_vel*Ymoy_ct[0,:,:],axis=1)

    if solver_param['hyper'] == True:

        flux[0 ,range_flux_right] = 0 
        flux[1 ,range_flux_right] = tau
        flux[2 ,range_flux_right] = (umoy * tau) - q
        flux[3:,range_flux_right] = (rmoy[:,np.newaxis]*mass_diff_mix[:,:-1]*dY_dx[:-1].T).T-(rmoy*corr_vel*Ymoy)

    else: 

        flux[0 ,1:-1] = 0 
        flux[1 ,1:-1] = tau
        flux[2 ,1:-1] = (umoy * tau) - q
        flux[3:,1:-1] = (rmoy[:,np.newaxis]*mass_diff_mix[:,:-1]*dY_dx[:-1].T).T-(rmoy*corr_vel*Ymoy)

        
    state['flux_visc_cons'] = flux

    return state

def second_order_roe_flux_calculator(solver_param,rom_param,state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)
    
        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']
        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]

        gamma  = solver_param['gamma']
        vol    = solver_param['vol']
        dx     = vol

        S_indx_user = rom_param['S_indx_user']

        # number of cell
        cell_num = len(rho)

        # breakpoint()
        # calculate gradient
        df_dx = (np.roll(Q_prim_user,-1) - np.roll(Q_prim_user,1)) / (2*dx)

        # limiter

        if solver_param['limiter']:

            df_dx = slope_limit(df_dx,Q_prim_user,dx,solver_param)

        # face_reconstruction
        face_prim_user = np.zeros((solver_param['num_prim_var'],2*(cell_num)+2))

        for indx in range(1,cell_num):

            face_prim_user[:,2*indx]   = Q_prim_user[:,indx-1] + df_dx[:,indx-1] * (dx/2)     # left face 
            face_prim_user[:,2*indx+1] = Q_prim_user[:,indx]   - df_dx[:,indx] * (dx/2)       # right face 

        rho   = face_prim_user[0,:]
        vx    = face_prim_user[1,:]
        press = face_prim_user[2,:]

        # total enthalpy
        # htot = gamma/(gamma-1)*press/rho+0.5*vx**2
        # breakpoint()
        flux = np.zeros((3,cell_num+1))
        diffusion = np.zeros((3,cell_num+1))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 

            range_flux_neighbor_left   = range_flux - 1
            range_flux_neighbor_right  = range_flux + 1 
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left,range_flux_neighbor_right))
            range_flux = np.sort(np.unique(range_flux))


        else : 
            
            range_flux = range(1,cell_num)

        for j in range_flux:
            en_L               = press[2*j]    / (gamma-1) + 0.5 * rho[2*j]    * (vx[2*j]**2)
            en_R               = press[2*j+1]  / (gamma-1) + 0.5 * rho[2*j+1]  * (vx[2*j+1]**2)

            # total enthalpy
            htot_L = gamma/(gamma-1)*press[2*j]/rho[2*j]+0.5*vx[2*j]**2
            htot_R = gamma/(gamma-1)*press[2*j+1]/rho[2*j+1]+0.5*vx[2*j+1]**2
        
            # Compute Roe averages
            R=np.sqrt(rho[2*j+1]/rho[2*j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                       # {hat rho}_{j+1/2}
            umoy=(R*vx[2*j+1]+vx[2*j])/(R+1)                    # {hat U}_{j+1/2}
            hmoy=(R*htot_R+htot_L)/(R+1);               # {hat H}_{j+1/2}
            amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy))  # {hat a}_{j+1/2}
            
            # Auxiliary variables used to compute P_{j+1/2}^{-1}
            alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
            alph2=(gamma-1)/(amoy*amoy)

            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                            [1-alph1,                alph2*umoy,                -alph2 ],
                            [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]])
                    
            # Compute matrix P_{j+1/2}
            P    = np.array([[ 1,              1,              1              ],
                            [umoy-amoy,        umoy,           umoy+amoy      ],
                            [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]])
            
            # Compute matrix Lambda_{j+1/2}
            lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                            [0,                 abs(umoy),      0                 ],
                            [0,                 0,              abs(umoy+amoy)    ]])
                        
            # Compute Roe matrix |A_{j+1/2}|
            A = P @ lamb @ Pinv

            diffusion[:,j] = 0.5 * A @ (Q_cons_user[:,j]-Q_cons_user[:,j-1]) / vol

            flux[0,j] = 0.5*(rho[2*j]*vx[2*j]                 + rho[2*j+1]*vx[2*j+1])                         - diffusion[0,j] 
            flux[1,j] = 0.5*(rho[2*j]*vx[2*j]**2 + press[2*j] + rho[2*j+1]*vx[2*j+1]**2+press[2*j+1])         - diffusion[1,j] 
            flux[2,j] = 0.5*(vx[2*j]*(en_L+press[2*j])        + vx[2*j+1]*(en_R+press[2*j+1]))                - diffusion[2,j] 


    ### If It is a Multi-Species Case ###

    else :
        
        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)

        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        vol    = solver_param['vol']
        dx     = vol

        # number of cell
        cell_num = len(Q_cons_user[0,:])

        # calculate gradient
        df_dx_prim = (np.roll(Q_prim_user,-1) - np.roll(Q_prim_user,1)) / (2*dx)
        df_dx_cons = (np.roll(Q_cons_user,-1) - np.roll(Q_cons_user,1)) / (2*dx)

        # limiter
        slope_limiter = solver_param['limiter']

        if slope_limiter:

            df_dx_prim = np.maximum(0., np.minimum(1., ( (Q_prim_user-np.roll(Q_prim_user,1,axis=0))/dx)/(df_dx_prim + 1.0e-8*(df_dx_prim==0)))) * df_dx_prim
            df_dx_prim = np.maximum(0., np.minimum(1., (-(Q_prim_user-np.roll(Q_prim_user,-1,axis=0))/dx)/(df_dx_prim + 1.0e-8*(df_dx_prim==0)))) * df_dx_prim

            df_dx_cons = np.maximum(0., np.minimum(1., ( (Q_cons_user-np.roll(Q_cons_user,1,axis=0))/dx)/(df_dx_cons + 1.0e-8*(df_dx_cons==0)))) * df_dx_cons
            df_dx_cons = np.maximum(0., np.minimum(1., (-(Q_cons_user-np.roll(Q_cons_user,-1,axis=0))/dx)/(df_dx_cons + 1.0e-8*(df_dx_cons==0)))) * df_dx_cons

        # face_reconstruction
        face_prim_user = np.zeros((solver_param['num_prim_var'],2*(cell_num)+2))
        face_cons_user = np.zeros((solver_param['num_state_var'],2*(cell_num)+2))

        for indx in range(1,cell_num):

            face_prim_user[:,2*indx]   = Q_prim_user[:,indx-1] + df_dx_prim[:,indx-1] * (dx/2)     # left face 
            face_prim_user[:,2*indx+1] = Q_prim_user[:,indx]   - df_dx_prim[:,indx] * (dx/2)       # right face   

            face_cons_user[:,2*indx]   = Q_cons_user[:,indx-1] + df_dx_cons[:,indx-1] * (dx/2)     # left face 
            face_cons_user[:,2*indx+1] = Q_cons_user[:,indx]   - df_dx_cons[:,indx] * (dx/2)       # right face         

        # # fill the ghost cells
        face_prim_user[:,0:2] = face_prim_user[:,2].reshape(-1,1)
        face_prim_user[:,-2:] = face_prim_user[:,-3].reshape(-1,1)

        face_cons_user[:,0:2] = face_cons_user[:,2].reshape(-1,1)
        face_cons_user[:,-2:] = face_cons_user[:,-3].reshape(-1,1)


        rho    = face_prim_user[0,:]
        vx     = face_prim_user[1,:]
        press  = face_prim_user[2,:]
        temp   = face_prim_user[3,:]


        Y      = face_prim_user[4:,:]
        Y_ct   = find_mass_fraction_full_cantera(Y)


        num_state_var = solver_param['num_state_var']
        num_species   = solver_param['num_species']

        S_indx_user = rom_param['S_indx_user']

        en               = face_cons_user[2,:] / vol

        state['gas_array_2nd_order'].TPY = temp,press,Y_ct

        c       = np.squeeze(state['gas_array_2nd_order'].sound_speed)
        int_en  = np.squeeze(state['gas_array_2nd_order'].int_energy_mass)
        h       = np.squeeze(state['gas_array_2nd_order'].enthalpy_mass)

        # total enthalpy
        htot = h + (0.5*(vx**2))

        flux = np.zeros((num_state_var,cell_num+1))
        diss_matrix = np.zeros((num_state_var,num_state_var))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))

        else : 
            
            range_flux = range(1,cell_num)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[2*j+1]/rho[2*j])                      # R_{j+1/2}
            rmoy=R*rho[2*j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[2*j+1]+vx[2*j])/(R+1)                    # {hat U}_{j+1/2}
            Hmoy=(R*htot[2*j+1]+htot[2*j])/(R+1)                # {hat H}_{j+1/2}
            emoy=(R*int_en[2*j+1]+int_en[2*j])/(R+1)              
            hmoy=Hmoy - (0.5*umoy*umoy)
            Ymoy=(R*Y[:,2*j+1]+Y[:,2*j])/(R+1)                  

            Ymoy_ct = find_mass_fraction_full_cantera(Ymoy.reshape(-1,1))

            state['gas'].UVY = emoy,1/rmoy,Ymoy_ct

            Pmoy         = state['gas'].P
            Tmoy         = state['gas'].T
            cpmoy        = state['gas'].cp
            cmoy         = state['gas'].sound_speed
            meanMWmoy    = state['gas'].mean_molecular_weight
            MWmoy        = state['gas'].molecular_weights
            partial_hmoy = state['gas'].partial_molar_enthalpies / MWmoy
            
            d_rho_d_press = rmoy / Pmoy
            d_rho_d_temp = -rmoy/Tmoy

            d_rho_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_rho_d_mass_frac[sp] = rmoy*meanMWmoy*(1/MWmoy[-1] - 1/MWmoy[sp])

            d_enth_d_press = 0
            d_enth_d_temp = cpmoy

            d_enth_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_enth_d_mass_frac[sp] = partial_hmoy[sp] - partial_hmoy[-1]

            # Gamma terms for energy equation
            g_press     = rmoy * d_enth_d_press + d_rho_d_press * Hmoy - 1.0
            g_temp      = rmoy * d_enth_d_temp + d_rho_d_temp * Hmoy
            g_mass_frac = rmoy * d_enth_d_mass_frac + Hmoy * d_rho_d_mass_frac

            # Characteristic speeds
            lambda1 = umoy + cmoy
            lambda2 = umoy - cmoy
            lambda1_abs = np.absolute(lambda1)
            lambda2_abs = np.absolute(lambda2)

            r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
            alpha = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
            beta  = np.power(cmoy, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
            phi   = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

            eta = (1.0 - rmoy * d_enth_d_press) / d_enth_d_temp
            psi = eta * d_rho_d_temp + rmoy * d_rho_d_press

            vel_abs = np.absolute(umoy)

            beta_star = beta * psi
            beta_e = beta * (rmoy * g_press + g_temp * eta)
            phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rmoy
            phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rmoy
            m = rmoy * alpha
            e = rmoy * umoy * alpha

            delta_p = press[2*j]-press[2*j+1]
            delta_u = vx[2*j]-vx[2*j+1]
            delta_T = temp[2*j]-temp[2*j+1]
            delta_Y = (Y[:,2*j]-Y[:,2*j+1]).reshape(-1,1)


            del_q_prim = np.vstack((delta_p,delta_u,delta_T,delta_Y))

            diss_matrix[ 0 , 0 ] = phi_star
            diss_matrix[ 0 , 1 ] = beta_star
            diss_matrix[ 0 , 2 ] = vel_abs * d_rho_d_temp
            diss_matrix[ 0 , 3:] = vel_abs * d_rho_d_mass_frac

            diss_matrix[ 1 , 0 ] = umoy * phi_star + r_roe
            diss_matrix[ 1 , 1 ] = umoy * beta_star + m
            diss_matrix[ 1 , 2 ] = umoy * vel_abs * d_rho_d_temp
            diss_matrix[ 1 , 3:] = (umoy * vel_abs) * d_rho_d_mass_frac

            diss_matrix[ 2 , 0 ] = phi_e + r_roe * umoy
            diss_matrix[ 2 , 1 ] = beta_e + e
            diss_matrix[ 2 , 2 ] = g_temp * vel_abs
            diss_matrix[ 2 , 3:] = g_mass_frac * vel_abs

            diss_matrix[3:, 0] = Ymoy * phi_star
            diss_matrix[3:, 1] = Ymoy * beta_star
            diss_matrix[3:, 2] = Ymoy * (vel_abs * d_rho_d_temp)

            for mf_idx_out in range(3, num_state_var):

                for mf_idx_in in range(3, num_state_var):

                    if mf_idx_out == mf_idx_in:
                        diss_matrix[mf_idx_out, mf_idx_in] = vel_abs * (
                            rmoy + Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )
                    else:
                        diss_matrix[mf_idx_out, mf_idx_in] = (
                            vel_abs * Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )

            dissipation = diss_matrix @ del_q_prim
        
            flux[0,j] = 0.5*(rho[2*j]*vx[2*j]                 + rho[2*j+1]*vx[2*j+1])                           + 0.5 * dissipation[0 ,0]
            flux[1,j] = 0.5*(rho[2*j]*vx[2*j]**2 + press[2*j] + rho[2*j+1]*vx[2*j+1]**2+press[2*j+1])           + 0.5 * dissipation[1 ,0]
            flux[2,j] = 0.5*(vx[2*j]*(en[2*j]+press[2*j])     + vx[2*j+1]*(en[2*j+1]+press[2*j+1]))             + 0.5 * dissipation[2 ,0]
            flux[3:,j]= 0.5*(rho[2*j]*vx[2*j]*Y[:,2*j]        + rho[2*j+1]*vx[2*j+1]*Y[:,2*j+1] )               + 0.5 * dissipation[3:,0]
    # breakpoint()

    # flux = np.hstack((flux[:,0].reshape(-1,1),flux[:,0].reshape(-1,1),flux,flux[:,-1].reshape(-1,1),flux[:,-1].reshape(-1,1))) 

    state['flux_cons'] = flux

    return state

def d_flux_dx_calculator_for(solver_param,rom_param,state):

    S_indx_user = rom_param['S_indx_user']

    if solver_param['viscous_flag']:
        
        flux = state['flux_cons']-state['flux_visc_cons']

    else:

        flux = state['flux_cons']

    if solver_param['hyper'] == True:

        d_flux_dx = np.zeros((solver_param['num_state_var'] , len(S_indx_user)))

        S_indx_user = S_indx_user + 2

        counter = 0

        for indx in S_indx_user:

            d_flux_dx[:,counter] = -(flux[:,indx+1] - flux[:,indx])
            counter = counter + 1

    else:

        d_flux_dx = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

        for indx in range(0,len(d_flux_dx[0,:])):

            d_flux_dx[:,indx] = -(flux[:,indx+1] - flux[:,indx])
    
    state['d_flux_dx'] = d_flux_dx.ravel()
    
    return state

def d_flux_dx_calculator(solver_param,rom_param,state):

    if solver_param['viscous_flag']:
        
        flux = state['flux_cons']-state['flux_visc_cons']

    else:

        flux = state['flux_cons']
    
    if solver_param['hyper'] == True:

        S_indx_user      = rom_param['S_indx_user']
        S_indx_user      = S_indx_user + 2
        range_flux_right = S_indx_user + 1
        range_flux_left  = S_indx_user

        d_flux_dx = -(flux[:, range_flux_right] - flux[:, range_flux_left])

    else:

        d_flux_dx = -(flux[:, 1:] - flux[:, :-1])
    
    state['d_flux_dx'] = d_flux_dx.ravel()
    
    return state

def results_solver2user_converter(num_state_var,num_cell,Q):

    Q_reshaped = np.reshape(Q,(num_state_var,num_cell+4))

    return Q_reshaped

def results_user2solver_converter(Q):

    Q_reshaped = np.ravel(Q)

    return Q_reshaped

def solver_eliminate_ghost(solver_param,Q_full):

    Q_full_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_full)
    Q_int_user = Q_full_user[:,2:-2]
    Q_int_solver = results_user2solver_converter(Q_int_user)

    return Q_int_solver

def solver_add_ghost(cell_number,num_var,Q_int):

    Q_int_user  = results_solver2user_converter(num_var,cell_number-4,Q_int)
    Q_int_full = np.column_stack((Q_int_user[:,0], Q_int_user[:,0], Q_int_user , Q_int_user[:,-1] , Q_int_user[:,-1]))
    Q_full_solver= results_user2solver_converter(Q_int_full)
    

    return Q_full_solver

def periodic_injection(solver_param,state):

    # read current states
    Q_prim      = state['Q_prim']
    Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    Q_cons      = state['Q_cons']
    Q_cons_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

    rho         = Q_prim_user[0,:]
    u           = Q_prim_user[1,:]    
    P           = Q_prim_user[2,:]
    T           = Q_prim_user[3,:]
    Y           = Q_prim_user[4:,:]
    Y_ct        = find_mass_fraction_full_cantera(Y)

    vol         = solver_param['vol']

    rho_in      = solver_param['injcetion_prim_state'][0]/2
    v_in        = solver_param['injcetion_prim_state'][1]
    P_in        = solver_param['injcetion_prim_state'][2]/2
    T_in        = solver_param['injcetion_prim_state'][3]
    Y_in        = solver_param['injcetion_prim_state'][4:]

    area_in     = solver_param['injector_face_area']

    state['gas'].TPY = T_in,P_in,Y_in 

    int_energy_in = state['gas'].int_energy_mass

    # read injection regions index
    add_indx    = state['injection_add_final']
    sub_indx    = state['injection_sub_init'] 

    indx_init   = np.argmax(P) - sub_indx
    indx_final  = indx_init + add_indx
    detonation  = np.arange(indx_init,indx_final)%(solver_param['cell_number']+4)
    inj_indx    = np.arange(0,solver_param['cell_number']+4,1)
    inj_indx    = inj_indx[~np.isin(inj_indx,detonation)]



    'Injection Method'
    inj_terms                       = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

    m_dot_in                        = rho_in * v_in * area_in / vol
    e_in_tot                        = int_energy_in + 0.5*u[inj_indx]**2

    inj_terms[0,inj_indx]           = m_dot_in
    # inj_terms[1,inj_indx]           = m_dot_in * u[inj_indx]
    # inj_terms[1,inj_indx]           = m_dot_in * 0.01
    inj_terms[1,inj_indx]           = -m_dot_in * v_in
    inj_terms[2,inj_indx]           = m_dot_in * e_in_tot
    inj_terms[3:,inj_indx]          = m_dot_in * Y_in[:-1]

    inj_terms = inj_terms * vol

    state['d_flux_dx'] = state['d_flux_dx'] + inj_terms.ravel()

    return state

def injection_correction(solver_param,state):

    # read current states
    Q_prim      = state['Q_prim']
    Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    Q_cons      = state['Q_cons']
    Q_cons_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

    rho         = Q_prim_user[0,:]
    u           = Q_prim_user[1,:]    
    P           = Q_prim_user[2,:]
    T           = Q_prim_user[3,:]
    Y           = Q_prim_user[4:,:]
    Y_ct        = find_mass_fraction_full_cantera(Y)

    vol         = solver_param['vol']

    # read injection regions index
    add_indx    = state['injection_add_final']
    sub_indx    = state['injection_sub_init'] 

    indx_init   = np.argmax(P) - sub_indx
    indx_final  = indx_init + add_indx
    detonation  = np.arange(indx_init,indx_final)%(solver_param['cell_number']+4)
    inj_indx    = np.arange(0,solver_param['cell_number']+4,1)
    inj_indx    = inj_indx[~np.isin(inj_indx,detonation)]

    # create mixed gas state
    gas_array_current  = ct.SolutionArray(state['gas'],(1,len(inj_indx)))
    gas_array_inject   = ct.SolutionArray(state['gas'],(1,len(inj_indx)))
    gas_array_mix      = ct.SolutionArray(state['gas'],(1,len(inj_indx)))

    # compute injection rate 
    rho_in      = solver_param['injcetion_prim_state'][0]
    v_in        = solver_param['injcetion_prim_state'][1]
    P_in        = solver_param['injcetion_prim_state'][2]
    T_in        = solver_param['injcetion_prim_state'][3]
    Y_in        = solver_param['injcetion_prim_state'][4:]

    area_in     = solver_param['injector_face_area']

    mass_current= rho * vol
    mass_in     = rho_in * v_in * area_in * solver_param['dt']

    total_mass  = mass_current+mass_in

    # species mass conservation
    partial_mass_current = mass_current    * Y_ct[0,:,:].T
    partial_mass_inject  = mass_in * Y_in
    Y_mix                = (partial_mass_current + partial_mass_inject) / total_mass

    # moment conservation
    # vx_mix       = (rho_in * 0.01 + rho * u)/(rho_in + rho)


    # energy conservation
    gas_array_current.TPY = T[inj_indx], P[inj_indx], Y_ct[0,inj_indx]
    gas_array_inject.TPY  = T_in,P_in,Y_in.T

    int_energy_current    = np.squeeze(gas_array_current.int_energy_mass)
    int_energy_in         = np.squeeze(gas_array_inject.int_energy_mass)

    int_energy_mix        = (mass_current[inj_indx] * int_energy_current + mass_in * int_energy_in)/total_mass[inj_indx]
    vol_inj               = mass_in / gas_array_inject.density
    total_volume          = vol + vol_inj
    sp_volume_mix         = total_volume / total_mass[inj_indx]

    gas_array_mix.UVY     = int_energy_mix,sp_volume_mix,Y_mix[:,inj_indx].T

    internal_energy       = np.squeeze(gas_array_mix.int_energy_mass)

    internal_energy_tot   = internal_energy + (0.5 * (u[inj_indx] **2))

    # mass conservation 
    rho_mix = np.squeeze(gas_array_mix.density)
    Y_mix   = gas_array_mix.Y[0,:,:-1].T

    # momentum conservation using isentropic
    gamma_current                    = np.squeeze(gas_array_current.cp/gas_array_current.cv)
    speed_sound_current              = np.squeeze(gas_array_current.sound_speed)
    mach_current                     = u[inj_indx]/speed_sound_current
    P_tot_current                    = P[inj_indx] * (1+(gamma_current-1)/2*mach_current**2)**((gamma_current-1)/gamma_current)
    P_tot_mix                        = P_tot_current

    P_mix                            = np.squeeze(gas_array_mix.P)
    speed_sound_mix                  = np.squeeze(gas_array_mix.sound_speed)
    gamma_mix                        = np.squeeze(gas_array_mix.cp/gas_array_mix.cv)
    pressure_ratio                   = P_tot_mix/P_mix
    pressure_ratio[pressure_ratio<1] = 1

    mach_mix                         = np.sqrt(2/(gamma_current-1)*((pressure_ratio)**((gamma_mix-1)/gamma_mix)-1))


    u_mix                            = mach_mix * speed_sound_mix

    # replace the influenced cells states
    mass         = rho_mix * vol
    momx         = rho_mix * u_mix * vol
    energy       = (rho_mix * internal_energy_tot) * vol
    mass_species = rho_mix * Y_mix * vol

    Q_cons_inject= np.vstack((mass,momx,energy,mass_species))

    Q_cons_user[:,inj_indx] = Q_cons_inject


    # Q_cons_user[:,detonation[0]-2:detonation[0]] = Q_cons_user[:,detonation[0]+1].reshape(-1,1)
    # # Create a simple averaging kernel (window size 11 in this case)
    # window_size = 5
    # kernel = np.ones(window_size)/window_size

    # # # Apply convolution along the time axis (axis=0 in this example)
    # smoothed_section = np.apply_along_axis(
    #     lambda m: np.convolve(m, kernel, mode='same'),
    #     axis=1,  # change this to 1 if you want to smooth along columns
    #     arr=Q_cons_user[:, detonation[0]-5:detonation[0]+5]
    # )

    # # Assign back to original array
    # Q_cons_user[:, detonation[0]-5:detonation[0]+5] = smoothed_section

    # smoothed_2d = np.zeros_like(Q_cons_user[:, detonation[0]-10:detonation[0]+10])

    # for i in range(12):

    #     smoothed_2d[i,:] = savgol_filter(Q_cons_user[i, detonation[0]-10:detonation[0]+10], window_length=20, polyorder=2)
    
    # Q_cons_user[:, detonation[0]-10:detonation[0]+10] = smoothed_2d

    smoothing_start_indx = (detonation[0]-10)%(solver_param['cell_number']+4)
    smoothing_end_indx   = (detonation[0]+10)%(solver_param['cell_number']+4)

    q_left  = Q_cons_user[:,smoothing_start_indx]
    q_right = Q_cons_user[:,smoothing_end_indx]

    x_knwon = np.array([0,20])
    y_known = np.array([q_left,q_right]).T

    f_interp = interp1d(x_knwon,y_known,kind='linear')

    x        = np.arange(0,20)

    q_ramp   = f_interp(x)

    if smoothing_start_indx < smoothing_end_indx:
        Q_cons_user[:, smoothing_start_indx:smoothing_end_indx] = q_ramp
    else:
        wrap_len = (solver_param['cell_number']+4) - smoothing_start_indx
        Q_cons_user[:, smoothing_start_indx:] = q_ramp[:, :wrap_len]
        Q_cons_user[:, :smoothing_end_indx]   = q_ramp[:, wrap_len:]

    state['Q_cons'] = Q_cons_user.ravel()

    # N = solver_param['cell_number'] + 4
    # smoothing_start_indx = (detonation[0]-10) % N
    # smoothing_end_indx   = (detonation[0]+10) % N

    # q_left  = Q_cons_user[:, smoothing_start_indx]
    # q_right = Q_cons_user[:, smoothing_end_indx]

    # x_knwon = np.array([0, 1])
    # y_known = np.array([q_left, q_right]).T
    # f_interp = interp1d(x_knwon, y_known, kind='linear')

    # x = np.linspace(0, 1, 20)
    # q_ramp = f_interp(x)

    # # Assign with wrap-around handling
    # if smoothing_start_indx < smoothing_end_indx:
    #     Q_cons_user[:, smoothing_start_indx:smoothing_end_indx] = q_ramp
    # else:
    #     wrap_len = N - smoothing_start_indx
    #     Q_cons_user[:, smoothing_start_indx:] = q_ramp[:, :wrap_len]
    #     Q_cons_user[:, :smoothing_end_indx]   = q_ramp[:, wrap_len:]

    #The original code for context
    # x = np.arange(detonation[0]-10,detonation[0]+10)%(solver_param['cell_number']+4)
    # q_left = Q_cons_user[:,x[0]]
    # q_right = Q_cons_user[:,x[-1]]
    # x_knwon = np.array([x[0],x[-1]])
    # y_known = np.array([q_left,q_right]).T
    # f_interp = interp1d(x_knwon,y_known,kind='linear')
    # q_ramp = f_interp(x)
    # Q_cons_user[:, detonation[0]-10:detonation[0]+10] = q_ramp

    return state

def find_mass_fraction_full(MF):
    MF_last_row = 1.0 - np.sum(MF, axis=0)
    MF_full = np.vstack((MF, MF_last_row))
    MF_full[MF_full == 0] = 1e-30
    return MF_full

def find_mass_fraction_full_cantera(MF):

    # this function first finds the full mass fraction matrix and 
    # then converts it to the shape that is suitable for cantera

    MF = find_mass_fraction_full(MF)

    # MF_shape = np.shape(MF)

    # # reshape the MF array suitable for cantera

    # MF_reshaped = np.zeros( (1,MF_shape[1],MF_shape[0]) )

    # for indx in range(0,MF_shape[0]):

    #     MF_reshaped[:,:,indx] = MF[indx,:]

    MF_reshaped = MF.T[np.newaxis, :, :]  # shape: (1, N_points, N_species)

    return MF_reshaped

def source_calculator(solver_param,rom_param,state):

    # load the basic information
    Q_cons           = state['Q_cons']
    # state            = cons2prim_converter(solver_param,state)
    Q_prim           = state['Q_prim']
    Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho              = Q_prim_user[0,:]
    vx               = Q_prim_user[1,:]
    press            = Q_prim_user[2,:]
    temp             = Q_prim_user[3,:]
    Y                = Q_prim_user[4:,:]

    Y_ct             = find_mass_fraction_full_cantera(Y)

    vol              = solver_param['vol']
    num_state_var    = solver_param['num_state_var']

    state['gas_array'].TPY = temp,press,Y_ct

    net_production_rate_ct = state['gas_array'].net_production_rates
    MW_species             = state['gas_array'].molecular_weights
    state['heat_release']  = np.squeeze(state['gas_array'].heat_release_rate)
    state['int_energy']    = np.squeeze(state['gas_array'].int_energy_mass)

    source_terms   = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

    source_terms[3:,:]   = (net_production_rate_ct[0,:,:-1] * MW_species[:-1] * vol).T

    state['source_terms'] = source_terms.ravel()

    if solver_param['hyper'] == True:

        source_terms_int = solver_eliminate_ghost(solver_param,state['source_terms'])

        state['source_terms'] = source_terms_int[rom_param['S_indx_solver']]

    state['d_flux_dx'] = state['d_flux_dx'] + state['source_terms']

    return state

def advance_one_time_step(solver_param,rom_param,state,fig,axs,visual_param):

    iter = solver_param['iter']

    if solver_param['solver_mode'] == 'FOM':

        state = residual_calculator(solver_param,rom_param,state)
        state = time_integrator_functions.advance_time(solver_param,rom_param,state)

    elif solver_param['solver_mode'] == 'ROM':
        
        state = residual_calculator(solver_param,rom_param,state)
        state , rom_param = rom_functions.modern_red2full_state_calculator(solver_param,rom_param,state)

    elif solver_param['solver_mode'] == 'Adaptive ROM':

        state, solver_param , rom_param  = rom_functions.adaptive_rom_progress(solver_param,rom_param,state,iter)

    elif solver_param['solver_mode'] == 'Hybrid ROM':

        if solver_param['rom_type'] == 'intrusive':

            state, solver_param , rom_param  = rom_functions.hybrid_rom_intrusive_progress(solver_param,rom_param,state,iter)

        elif solver_param['rom_type'] == 'non-intrusive':

            state, solver_param , rom_param  = rom_functions.hybrid_rom_non_intrusive_progress(solver_param,rom_param,state,iter)

    if solver_param['injection']:

        state = injection_correction(solver_param,state)

    # Q_cons_user = solver_functions.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],[state['Q_cons']])
    # Q_cons_user[:,0:3] = Q_cons_user[:,4].reshape(-1,1)
    # Q_cons_user[:,-3:] = Q_cons_user[:,-4].reshape(-1,1)

    # convert cons to prim
    state = cons2prim_converter(solver_param,state)

    # update the ghost cells
    state = update_ghost_cell(solver_param,state)

    state['time'] = state['time'] + solver_param['dt']

    if iter < 1:

        state['Q_cons_old'] = solver_add_ghost(solver_param['cell_number'],
                                                                solver_param['num_state_var'],
                                                                state['cons_results_save'])

    # prepare data to save
    state['cons_results_save'] = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],[state['Q_cons']])[:,2:-2]

    if solver_param['solver_mode'] == 'FOM': 

        state['res_save']          = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['d_flux_dx'])[:,2:-2]

    else:

        state['res_save']          = np.zeros(solver_param['num_state_var']*solver_param['cell_number'])

        if len(rom_param['S_indx_solver']) != len(state['d_flux_dx']):

            state['res_save'][rom_param['S_indx_solver']] = solver_eliminate_ghost(solver_param,state['d_flux_dx'])[rom_param['S_indx_solver']]
        else:

            state['res_save'][rom_param['S_indx_solver']] = state['d_flux_dx']
        
    if solver_param['gas_model'] == 'Non-Reacting Air':

        state['prim_results_save'] = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

    else :
        
        state['prim_results_save'][:-1,:] = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
        state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]

    # save the data 

    if iter % solver_param['save_interval'] == 0:

        results_recorder(solver_param,rom_param,state)

    if solver_param['visual'] == True:


        # visualization
        visualization_functions.in_progress_plot(fig,axs,iter,solver_param,rom_param,state,visual_param)
        
        plt.show(block=False)


    print('Iteration: ' + str(iter))

    return solver_param,rom_param,state,fig,axs,visual_param

def residual_calculator(solver_param,rom_param,state):
    
    if solver_param['flux_scheme'] == '1st Order Roe':

        state = first_order_roe_inviscid_flux_calculator(solver_param,rom_param,state)

        if solver_param['viscous_flag']:

            state = first_order_roe_viscous_flux_calculator(solver_param,rom_param,state)

    elif solver_param['flux_scheme'] == '2nd Order Roe':

        state = second_order_roe_flux_calculator(solver_param,rom_param,state)

        if solver_param['viscous_flag']:

            state = first_order_roe_viscous_flux_calculator(solver_param,rom_param,state)

    # inviscid flux vector
    state   = d_flux_dx_calculator(solver_param,rom_param,state)

    if solver_param['gas_model'] != 'Non-Reacting Air':

        state = source_calculator(solver_param,rom_param,state)

        # if solver_param['injection']:

        #     state = periodic_injection(solver_param,state)

    return state




    

