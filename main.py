# ============================================================================
# Reduce order model for particle pair sedimentation
# Author : Lucka Barbeau, Polytechnique Montréal, 2023

# MAIN PROGRAM
# ============================================================================


import matplotlib.pyplot as plt
import numpy as np
from particle_pair_rom import *
 

tf.random.set_seed(
    42
)


# Initialize the model
rom_model=ROM_2_particles(final_time=0.4,time_step=0.00025)

#Setup the case 
# units for length (millimeters ), units for time (second), units for mass (Kilograms)


rom_model.gravity=np.array([0,0,-10000])
rom_model.fluid.rho=0.000001
rom_model.fluid.mu=0.00070035 
rom_model.particle1.position=np.array([0.0,0.0,0])
rom_model.particle2.position=np.array([0.0,0.5,2.0])
rom_model.particle1.rho=0.001
rom_model.particle2.rho=0.001
rom_model.particle1.diameter=1.0
rom_model.particle2.diameter=1.0

#Update the particle state with the variables
rom_model.particle1.update()
rom_model.particle2.update()

# Run the case
output=rom_model.run()

# Graph results
fig1 = plt.figure()
ax1 = fig1.add_subplot() 
ax1.plot(output.time_table,output.vectorize_components(output.p1_velocity,1),label=r"$P_0$ ROM", color='k')
ax1.plot(output.time_table,output.vectorize_components(output.p2_velocity,1),label=r"$P_1$ ROM", color='b')
ax1.set_xlabel(r"t (s)")
ax1.set_ylabel(r"$v_y$ $\frac{mm}{s}$")
ax1.legend()
fig1.tight_layout()


fig2 = plt.figure()
ax2 = fig2.add_subplot() 
ax2.plot(output.time_table,output.vectorize_components(output.p1_velocity,2),label=r"$P_0$ ROM", color='k')
ax2.plot(output.time_table,output.vectorize_components(output.p2_velocity,2),label=r"$P_1$ ROM", color='b')
ax2.set_xlabel(r"t (s)")
ax2.set_ylabel(r"$v_z$ $\frac{mm}{s}$")

ax2.legend()
fig2.tight_layout()    

plt.show()



