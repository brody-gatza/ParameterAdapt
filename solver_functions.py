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

    return solver_param

def ic_generator(solver_param):

    # rho = np.linspace(solver_param['rho_inlet'] , solver_param['rho_outlet'] , int(solver_param['cell_number'])+2)
    # vx  = np.linspace(solver_param['vel_inlet'] , solver_param['vel_outlet'] , int(solver_param['cell_number'])+2)   
    # P   = np.linspace(solver_param['press_inlet'] , solver_param['press_outlet'] , int(solver_param['cell_number'])+2)

    rho = np.zeros((int(solver_param['cell_number'])+2))
    vx  = np.zeros((int(solver_param['cell_number'])+2))   
    P   = np.zeros((int(solver_param['cell_number'])+2))

    rho [0:257] = 1
    vx [0:257] = 0
    P [0:257] = 1

    rho [257:] = 0.125
    vx [257:] = 0
    P [257:] = 0.1

    
    return rho,vx,P

def add_ghost_cell(rho , vx , P):

    rho[0] = rho[1]
    vx[0]  = vx[1]
    P[0]   = P[1]

    rho[-1] = rho[-2]
    vx[-1]  = vx[-2]
    P[-1]   = P[-2]

    return rho,vx,P
    

def prim2cons_converter(rho , vx , P , gamma , vol):

    mass   = rho * vol
    momx   =  rho * vx * vol
    energy = (P/(gamma-1) + 0.5 * rho * (vx**2)) * vol

    return mass , momx , energy

def cons2prim_converter(mass , momx , energy , gamma , vol):

    rho = mass / vol
    vx  = momx / rho / vol
    P   = (energy/vol - (0.5 * rho * (vx**2))) * (gamma-1)

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
