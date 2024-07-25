import sys
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

gas = ct.Solution('gri30.yaml')

Temp=3000
Press=ct.one_atm
X=  'CH4:1.0, O2:1.0, N2:3.76'

gas.TPX = Temp, Press, X

for i in [ 1,  2,  3,  6, 12, 13, 34, 37, 47]:
    # gas.net_production_rates[i]
    # gas.species_name[i]
    print('new prod rate of ' + str(gas.species_names[i]) + ' is ' + str(gas.net_production_rates[i]))

# np.where(gas.net_production_rates!=0)



# gas.equilibrate('TP')


# gas.net_progress_rates

