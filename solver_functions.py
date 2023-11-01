import numpy as np

def solver_parameters_collector(self):

    solver_param = {}

    ### time discretization ###
    solver_param['dt'] = float(self.dt_entry_var.get())
    solver_param['num_step'] = int(self.num_step_entry_var.get())
    solver_param['time_scheme'] = self.time_scheme_entry_var.get()
    solver_param['dual_time'] = self.dt_entry_var.get()
    solver_param['res_tol'] = float(self.dt_entry_var.get())

    ### space discretization ###
    solver_param['x_initial'] = float(self.mesh_x_init_entry_var.get())
    solver_param['x_final'] = float(self.mesh_x_final_entry_var.get())
    solver_param['cell_number'] = int(self.cell_number_entry_var.get())

    ### inlet BC ###
    solver_param['press_inlet'] = float(self.inlet_press_entry_var.get())
    solver_param['temp_inlet'] = float(self.inlet_temp_entry_var.get())
    solver_param['vel_inlet'] = float(self.inlet_vel_entry_var.get())
    solver_param['rho_inlet'] = float(self.inlet_rho_entry_var.get())
    solver_param['mass_frac_inlet'] = self.inlet_mass_frac_entry_var.get()

    ### outlet BC ###
    solver_param['press_outlet'] = float(self.outlet_press_entry_var.get())
    solver_param['temp_outlet'] = float(self.outlet_temp_entry_var.get())
    solver_param['vel_outlet'] = float(self.outlet_vel_entry_var.get())
    solver_param['rho_outlet'] = float(self.outlet_rho_entry_var.get())
    solver_param['mass_frac_outlet'] = self.outlet_mass_frac_entry_var.get()

    ### gas model ###
    solver_param['gas_model'] = self.gas_model_entry_var.get()

    ### ROM ###
    solver_param['calc_rom'] = self.rom_method_checkbox_check_var.get()
    solver_param['rom_method'] = self.rom_method_entry_var.get()
    solver_param['pod_energy'] = float(self.energy_capture_entry_var.get())

    ### Hyper-Reduction ###
    solver_param['hyper'] = self.hyper_method_checkbox_check_var.get()
    solver_param['hyper_method'] = self.hyper_method_entry_var.get()

    ### visualization ###
    solver_param['variable1'] = self.visual_1_option_entry_var.get()
    solver_param['variable2'] = self.visual_2_option_entry_var.get()
    solver_param['variable3'] = self.visual_3_option_entry_var.get()
    solver_param['variable4'] = self.visual_4_option_entry_var.get()

    ### ic data ###
    solver_param['ic_data']   = self.ic_data

    ### Input Directory ###
    solver_param['working_dir'] = self.working_dir_entry_var.get()
    # solver_param['FOM_Result'] = self.FOM_re

    return solver_param

# def ic_generator(solver_param):

#     num_region = len(solver_param['ic_data'])
#     num_cell   = solver_param['cell_number']
#     x       =  np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number'])+4 )

#     rho = np.zeros(num_cell+4)
#     vx  = np.zeros(num_cell+4)
#     P   = np.zeros(num_cell+4)

#     for region in range(0,num_region):
        
#         indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )
        
#         rho[indx] = eval(solver_param['ic_data'][region][5])
#         vx [indx] = eval(solver_param['ic_data'][region][4])
#         P  [indx] = eval(solver_param['ic_data'][region][2])
         
#     return rho,vx,P

def ic_generator(solver_param):

    num_region = len(solver_param['ic_data'])
    num_cell   = solver_param['cell_number']
    x          =  np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number'])+4 )

    rho = np.zeros(num_cell+4)+0.01
    vx  = np.zeros(num_cell+4)+1
    P   = np.zeros(num_cell+4)+1

    rho = (1 * np.exp(-((x - 0.01/2) ** 2) / (2 * 0.001 ** 2)))+0.05

    return rho,vx,P

def add_ghost_cell(rho , vx , P ,sol_time):

    # rho[0:2]   = rho[3]
    # vx[0:2]    = vx[3]
    # P[0:2]     = P[3]

    # rho[-2:-1] = rho[-3]
    # vx[-2:-1]  = vx[-3]
    # P[-2:-1]   = P[-3]

    
    rho[0:2]   = rho[-4:-2]
    vx[0:2]    = vx[-4:-2]
    P[0:2]     = P[-4:-2]

    rho[-2:] = rho[2:4]
    vx[-2:]  = vx[2:4]
    P[-2:]   = P[2:4]

    rho[1]   = rho[-3]
    vx[1]    = vx[-3]
    P[1]     = P[-3]

    rho[-2] = rho[2]
    vx[-2]  = vx[2]
    P[-2]   = P[2]

    # if periodic_inlet:
         
    #     pass
         
    # if periodic_outlet:
         
    # pert =  0.0001* np.sin(2*np.pi*1e6*sol_time)
    # P[-3] = P[-3] * (1 + pert)
 
    return rho,vx,P
    

def prim2cons_converter(rho , vx , P , gamma , vol):

    mass   = rho * vol
    momx   =  rho * vx * vol
    energy = (P/(gamma-1) + 0.5 * rho * (vx**2)) * vol

    return mass , momx , energy

def cons2prim_converter(mass , momx , energy , gamma , vol,sol_time):

    rho = mass / vol
    vx  = momx / rho / vol
    P   = (energy/vol - (0.5 * rho * (vx**2))) * (gamma-1)

    rho , vx , P = add_ghost_cell(rho,vx,P,sol_time)

    return  rho , vx , P

def gradient_calculator(var,dx):

    df_dx = (np.roll(var,-1) - np.roll(var,1)) / (2*dx)

    return df_dx

def slope_limit(f, dx, df_dx):

	df_dx = np.maximum(0., np.minimum(1., ( (f-np.roll(f,1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx
	df_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,-1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx

	return df_dx

def extrapolate_center2face( var , d_var , dx):

    var_left_face  = var - (d_var * dx/2)
    var_left_face  = np.roll(var_left_face,-1)
    var_right_face = var + (d_var * dx/2)

    return var_left_face , var_right_face

def flux_calculator(rho_face_left , rho_face_right , vx_face_left , vx_face_right , p_face_left , p_face_right , gamma):

    en_L = p_face_left  / (gamma-1) + 0.5 * rho_face_left  * (vx_face_left**2)
    en_R = p_face_right / (gamma-1) + 0.5 * rho_face_right * (vx_face_right**2)

	# compute star (averaged) states
    rho_star  = 0.5*(rho_face_left + rho_face_right)
    momx_star = 0.5*(rho_face_left * vx_face_left + rho_face_right * vx_face_right)
    en_star   = 0.5*(en_L + en_R)
	
    P_star = (gamma-1)*(en_star-0.5*(momx_star**2)/rho_star)
	
	# compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_mass   = momx_star
    flux_momx   = momx_star**2/rho_star + P_star
    flux_energy = (en_star+P_star) * momx_star/rho_star
	
	# find wavespeeds
    C_L = np.sqrt(gamma * p_face_left  / rho_face_left  ) + np.abs(vx_face_left)
    C_R = np.sqrt(gamma * p_face_right / rho_face_right ) + np.abs(vx_face_right)
    C   = np.maximum( C_L, C_R )
	
	# add stabilizing diffusive term
    flux_mass   -= C * 0.5 * (rho_face_left - rho_face_right)
    flux_momx   -= C * 0.5 * (rho_face_left * vx_face_left - rho_face_right * vx_face_right)
    flux_energy -= C * 0.5 * ( en_L - en_R )

    return flux_mass, flux_momx, flux_energy

def inviscid_d_flux_dx_calculator(flux , dx):
     
    d_flux_dx = ( flux - np.roll(flux,1) ) / dx

    return d_flux_dx
