def explicit_rk4( var , ode , dt ):

    k1 = dt * ode
    k2 = (dt/2) * ode
    k3 = (dt/2) * ode
    k4 = dt * ode
    d_var = ( k1 + 2*k2 + 2*k3 + k4 ) / 6
    var = var + d_var

    # d_var = dt * ode 
    # var =  var + d_var

    return var