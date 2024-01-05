import solver_functions
import numpy as np

def flux_calculator(mass,momx,energy,dx,gamma,vol,slope_limiter,sol_time,S_indx_user,hyper_flag,flux_scheme,pcc):
    
    # compute flux
    if flux_scheme == 'Rusanov':
        flux_mass, flux_momx, flux_energy = solver_functions.rusanov_flux_calculator(mass , momx , energy ,gamma , vol , dx , slope_limiter , sol_time)

    elif flux_scheme == 'Roe':
        flux_mass, flux_momx, flux_energy = solver_functions.roe_flux_calculator(mass , momx , energy , gamma , vol , S_indx_user , hyper_flag , pcc)


    # inviscid flux vector terms
    dflux_mass_dx   = solver_functions.inviscid_d_flux_dx_calculator( flux_mass   , dx , hyper_flag , S_indx_user)
    dflux_momx_dx   = solver_functions.inviscid_d_flux_dx_calculator( flux_momx   , dx , hyper_flag , S_indx_user)
    dflux_energy_dx = solver_functions.inviscid_d_flux_dx_calculator( flux_energy , dx , hyper_flag , S_indx_user)

    return dflux_mass_dx , dflux_momx_dx , dflux_energy_dx
