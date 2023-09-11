# -*- coding: utf-8 -*-
# ============================================================================
# Reduce order model for particle pair sedimentation
# Author : Lucka Barbeau, Polytechnique Montréal, 2023

# ROM Class definition
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import joblib as joblib
import tensorflow as tf

tf.random.set_seed(
    42
)
# Huge speed increase in the evaluation of the model
tf.compat.v1.disable_eager_execution()


class ROM_2_particles():
    """Class that implements the reduced order model for the sedimentation of two particles.
    """

    def __init__(self,final_time=1.0,time_step=0.001):
        """Initialized the class

        Args:
            final_time: Variable that defines the end time of the simulation
            time_step: Variable that defines the time step of the simulation
        """
        self.dt=time_step
        self.t=0
        self.final_time=final_time
        # Load forces and torque models. The file containing the model must be in the same directory as this file.
        self.model_cd = keras.models.load_model('drag_model')
        self.model_cl = keras.models.load_model('lift_model')
        self.model_ct = keras.models.load_model('torque_model')
        self.scaler_x_cd = joblib.load("drag_model/scalerx.save") 
        self.scaler_y_cd = joblib.load("drag_model/scalery.save") 
        self.scaler_x_cl = joblib.load("lift_model/scalerx.save") 
        self.scaler_y_cl = joblib.load("lift_model/scalery.save") 
        self.scaler_x_ct = joblib.load("torque_model/scalerx.save") 
        self.scaler_y_ct = joblib.load("torque_model/scalery.save")
        # Initialized the object associated with each of the particles and the fluid. These can be modified before launching the simulation.
        self.particle1= self.particle(p_id=1)
        self.particle2= self.particle(p_id=2)
        self.fluid=self.fluid_properties()
        self.gravity=np.array([0.0,0.0,-9.810])
    
    def reset_to_initial_state(self):
        """A function that resets the history of the particle if the same object is used to do multiple simulations.
        """ 
        self.particle1.previous_velocity_list=[]
        self.particle2.previous_velocity_list=[]
        self.particle1.previous_basset_list=[]
        self.particle2.previous_basset_list=[]
        self.particle1.previous_omega_list=[]
        self.particle2.previous_omega_list=[]
        self.particle1.is_in_contact=False
        self.particle2.is_in_contact=False
        self.particle1.velocity=np.array([0.0,0.0,0.0])
        self.particle2.velocity=np.array([0.0,0.0,0.0])
        self.particle1.omega=np.array([0.0,0.0,0.0])
        self.particle2.omega=np.array([0.0,0.0,0.0])
        self.particle1.last_velocity=np.copy(self.particle1.velocity)
        self.particle2.last_velocity=np.copy(self.particle2.velocity)
        self.particle1.last_position=np.copy(self.particle1.position)
        self.particle2.last_position=np.copy(self.particle2.position)
        self.particle1.last_omega=np.copy(self.particle1.omega)
        self.particle2.last_omega=np.copy(self.particle2.omega)
        self.t=0


    class particle():
        """A class that implements the definition of a particle and its properties.
        """
        def __init__(self, diameter=1.0,rho=1.0 , position=np.array([0.0,0.0,0.0]), velocity=np.array([0.0,0.0,0.0]), omega=np.array([0.0,0.0,0.0]),p_id=0):
            """Initialized the class

            Args:
                diameter: Variable that defines the paritcle diameter
                rho: Variable that defines the density of the particle
                position (array of size 3): Variable that defines the initial position of the particle
                velocity (array of size 3): Variable that defines the initial velocity of the particle
                omega (array of size 3): Variable that defines the initial angular velocity of the particle
            """
            self.diameter=diameter
            self.position=position
            self.velocity=velocity
            self.omega=omega
            self.last_velocity=np.copy(velocity)
            self.last_position=np.copy(position) 
            self.last_omega=np.copy(omega)
            self.previous_velocity_list=[]
            self.previous_omega_list=[]
            self.previous_basset_list=[]
            self.last_M=np.zeros((3,3))
            self.last_M_ind=np.zeros((3,3))
            self.mass=diameter**3/6*np.pi*rho
            self.rho=rho
            self.id=p_id
            self.inertia=2/5*self.mass*(diameter/2)**2
            self.fd_norm=0
            self.ftotal_norm=0
            self.friction_coef=0
            self.restitution_coef=1
            self.young_modulus=1000
            self.tengential_overlap=np.copy(position)*0
            self.contact_velocty=np.copy(velocity)*0
            self.is_in_contact=False
            self.terminal_velocity=1

        def update(self):
            """A function that updates the particle's properties if its density or diameter has been changed.
            """ 
            self.mass=self.diameter**3/6*np.pi*self.rho
            self.inertia=2/5*self.mass*(self.diameter/2)**2
   

            
    class fluid_properties:
        """A class that implements the definition of the fluid and its properties.
        """
        def __init__(self, rho=1.0, mu=1.0):
            self.rho=rho
            self.mu=mu

    class output:
            """A class that implements the output object of the simulation. This class aims at facilitating the postprocess of a simulation by regrouping all the output arrays in one object.
            """
            def __init__(self,time_table,p1_position,p2_position,p1_velocity,p2_velocity,p1_omega,p2_omega,p1_total_fluid_force,p2_total_fluid_force,p1_total_fluid_torque,p2_total_fluid_torque,
                p1_virtual_mass_force,p2_virtual_mass_force,p1_meshchersky_force,p2_meshchersky_force,p1_drag_force,p2_drag_force,p1_lift_force,p2_lift_force,p1_lubrication_force,p2_lubrication_force,
                p1_magnus_force,p2_magnus_force,p1_history_force,p2_history_force,p1_contact_force,p2_contact_force,p1_buoyancy_force,p2_buoyancy_force,p1_induce_torque,
                p2_induce_torque,p1_viscous_torque,p2_viscous_torque,p1_history_torque,p2_history_torque,p1_contact_torque,p2_contact_torque,p1_lubrication_torque,p2_lubrication_torque):
                """Initialized the output object by passing it all the output variables.

                Args:
                    array of all the output variables.
                """

                self.time_table=time_table

                self.p1_position=p1_position
                self.p2_position=p2_position
                self.p1_velocity=p1_velocity
                self.p2_velocity=p2_velocity
                self.p1_omega=p1_omega
                self.p2_omega=p2_omega
                self.p1_total_fluid_force=p1_total_fluid_force
                self.p2_total_fluid_force=p2_total_fluid_force
                self.p1_total_fluid_torque=p1_total_fluid_torque
                self.p2_total_fluid_torque=p2_total_fluid_torque

                self.p1_virtual_mass_force=p1_virtual_mass_force
                self.p2_virtual_mass_force=p2_virtual_mass_force
                self.p1_meshchersky_force=p1_meshchersky_force
                self.p2_meshchersky_force=p2_meshchersky_force
                self.p1_drag_force=p1_drag_force
                self.p2_drag_force=p2_drag_force
                self.p1_lift_force=p1_lift_force
                self.p2_lift_force=p2_lift_force
                self.p1_lubrication_force=p1_lubrication_force
                self.p2_lubrication_force=p2_lubrication_force
                self.p1_magnus_force=p1_magnus_force
                self.p2_magnus_force=p2_magnus_force
                self.p1_history_force=p1_history_force
                self.p2_history_force=p2_history_force
                self.p1_contact_force=p1_contact_force
                self.p2_contact_force=p2_contact_force
                self.p1_buoyancy_force=p1_buoyancy_force
                self.p2_buoyancy_force=p2_buoyancy_force
                
                self.p1_induce_torque=p1_induce_torque
                self.p2_induce_torque=p2_induce_torque
                self.p1_viscous_torque=p1_viscous_torque
                self.p2_viscous_torque=p2_viscous_torque
                self.p1_history_torque=p1_history_torque
                self.p2_history_torque=p2_history_torque
                self.p1_contact_torque=p1_contact_torque
                self.p2_contact_torque=p2_contact_torque
                self.p1_lubrication_torque=p1_lubrication_torque
                self.p2_lubrication_torque=p2_lubrication_torque

            def vectorize_components(self,variable,component_index):
                """ A function that extracts and create a vector of one of the components of the output variables

                Args:
                    variable (n by 3 array): The output result of one of the variables
                    component_index : the index of the component we want to extract 0 for x component, 1 for the y component, 2 for the z component
                Returns:
                    component_vector: the vector for a given component of the output vector given in the input.
                """
                component_vector=np.zeros(len(variable))
                for i in range(len(variable)):
                    component_vector[i]=variable[i][component_index]
                return component_vector
                

    def terminal_velocity(self,particle1):
        """ A function that return the terminal velocity of a given particle alone.

        Args:
            particle1: the particle for which we calculate the terminal velocity.
        Returns:
            vf : The terminal velocity of the particle.
        """

        g=np.linalg.norm(self.gravity)
        vf=1
        residual_i=4.0/3.0*abs(particle1.rho-self.fluid.rho)*g*particle1.diameter-vf**2 *(self.fluid.rho*self.Cd_0(vf*particle1.diameter*self.fluid.rho/self.fluid.mu))
        last_residual=np.copy(residual_i)
        last_vf=np.copy(vf)
        vf=vf+0.01
        while abs(residual_i)>1e-6:
            residual_i=4.0/3.0*abs(particle1.rho-self.fluid.rho)*g*particle1.diameter-vf**2 *(self.fluid.rho*self.Cd_0(vf*particle1.diameter*self.fluid.rho/self.fluid.mu))
            dvf=-residual_i*(vf-last_vf)/(residual_i-last_residual)
            last_vf=np.copy(vf)
            last_residual=np.copy(residual_i)
            vf=vf+dvf
        return vf

    def stokes_time(self):
        """ A function that return the Stokes time for for particle 1 assuming the velocity is its terminal velocity. 
            This function calculate the general Stokes number proposed by Israel & Rosner. https://doi.org/10.1080/02786828308958612

        Args:
            particle1: the particle for which we calculate the terminal velocity.
        Returns:
            stokes time : The terminal velocity of the particle.
        """
        vf=self.terminal_velocity(self.particle1)
        Re=np.linalg.norm(vf)*self.particle1.diameter*self.fluid.rho/self.fluid.mu
        t0=(self.particle1.rho+0.5*self.fluid.rho)*self.particle1.diameter**2/18.0/self.fluid.mu

        if(Re>20):
            dre=0.01;
            int=0.501847
            re_i=20
            while re_i<Re-dre:
                int+=dre*(1/(self.Cd_0(re_i)*re_i)+1/(self.Cd_0(re_i+dre)*(re_i+dre)))/2
                re_i+=dre
            correction=24/Re*int
        else:
            dre=0.01;
            int=0
            re_i=0
            while re_i<Re-dre:
                int+=dre*(1/(24*(1+0.1315*re_i**(0.82-0.05*np.log10(re_i))))+1/(24*(1+0.1315*(re_i+dre)**(0.82-0.05*np.log10(re_i+dre)))))/2
                re_i+=dre
            correction=24/Re*int
            
        return t0*correction
    
    def stoke_number(self):
        """ A function that returns the Stokes number of particle 1 
            This function calculates the general Stokes number proposed by Israel & Rosner. https://doi.org/10.1080/02786828308958612
        Args:
            particle1: the particle for which we calculate the terminal velocity.
        Returns:
            the Stokes number: The stokes number of particle 1.
        """
        time=self.stokes_time()
        vf=self.terminal_velocity(self.particle1)
        return time*vf/self.particle1.diameter


    def D_ij(self,particle1,particle2):
        """ Calculates the relative position vector going from particle 1 to particle 2
            Note particle1 here is not the same as self.paritcle1 (same for particle2). 
            This allows the same function to determine the force on both particles by changing the reference particle in the calculation.
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
        Returns:
            Relative position vector from particle 1 to particle 2
        """
        return particle2.position-particle1.position

    def delta_tensor(self,particle1,particle2):         
        """ Calculates the relative position vector going from particle 1 to particle tensor. This is used in the virtual mass force calculation.
            Note particle1 here is not the same as self.paritcle1 (same for particle2). This allows the same function to determine the force on both particles by changing the reference particle in the calculation.
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
        Returns:
            Relative position vector from particle 1 to particle 2
        """
        D=self.D_ij(particle1,particle2)
        D_norm=1.0/np.linalg.norm(D)**2
        delta=np.zeros(3)
        delta[0]=(D[0])
        delta[1]=(D[1])
        delta[2]=(D[2])
        delta_t=np.zeros((3,3))
        for i in range(3) :
            for j in range(3) :
                delta_t[i][j]=delta[i]*delta[j]
        return delta_t*D_norm


    # coefficients function
    def Cm_pe_pe(self,particle1,D):
        """ Calculates the perpenticalar motion virtual mass coefficient
            Note particle1 here is not the same as self.paritcle1 (same for particle2). This allows the same function to determine the force on both particles by changing the reference particle in the calculation.
        Args:
            particle1: the reference particle in the calculation. Usually, the particle for which we are currently calculating the force. 
            D: the relative position vecotr
        Returns:
            cm_pe_pe: the perpendicular motion virtual mass coefficient
        """
        D_norm=np.linalg.norm(D)
        return 0.5*(1+3.0/256*(particle1.diameter/ D_norm)**6+3.0/256*(particle1.diameter/ D_norm)**8+27.0/4096*(particle1.diameter/ D_norm)**10)

    def Cm_pa_pa(self,particle1,D):
        """ Calculates the parallel motion virtual mass coefficient
            Note particle1 here is not the same as self.paritcle1 (same for particle2). This allows the same function to determine the force on both particles by changing the reference particle in the calculation.
        Args:
            particle1: the reference particle in the calculation. Usually, the particle for which we are currently calculating the force. 
            D: the relative position vecotr
        Returns:
            cm_pe_pe: the parallel motion virtual mass coefficient
        """
        D_norm=np.linalg.norm(D)
        return 0.5*(1+3.0/64*(particle1.diameter/ D_norm)**6+9.0/256*(particle1.diameter/ D_norm)**8+9.0/512*(particle1.diameter/ D_norm)**10)

    def Cm_ind_pe_pe(self,particle1,D):
        """ Calculates the induce perpenticalar  motion virtual mass coefficient
            Note particle1 here is not the same as self.paritcle1 (same for particle2). This allows the same function to determine the force on both particles by changing the reference particle in the calculation.
        Args:
            particle1: the reference particle in the calculation. Usually, the particle for which we are currently calculating the force. 
            D: the relative position vecotr
        Returns:
            cm_ind_pe_pe: the induce perpenticalar motion virtual mass coefficient
        """
        D_norm=np.linalg.norm(D)
        return 0.5*(3.0/16*(particle1.diameter/ D_norm)**3+3.0/4096*(particle1.diameter/ D_norm)**9+3.0/2048*(particle1.diameter/ D_norm)**11)

    def Cm_ind_pa_pa(self,particle1,D):
        """ Calculates the induce parallel motion virtual mass coefficient
            Note particle1 here is not the same as self.paritcle1 (same for particle2). This allows the same function to determine the force on both particles by changing the reference particle in the calculation.
        Args:
            particle1: the reference particle in the calculation. Usually, the particle for which we are currently calculating the force. 
            D: the relative position vecotr
        Returns:
            cm_ind_pa_pa: the induce parallel motion virtual mass coefficient
        """
        D_norm=np.linalg.norm(D)
        return 0.5*(-3.0/8*(particle1.diameter/ D_norm)**3-3.0/512*(particle1.diameter/ D_norm)**9-9.0/1024*(particle1.diameter/ D_norm)**11)

    #matrix virtual mass
    def M_matrix(self,particle1,particle2,fluid):
        """ Calculates the virtual mass matrix
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
            fluid: the fluid object.
        Returns:
           M: the virtual mass matrix
        """
        M=np.zeros((3,3))
        I=np.identity(3)
        D=self.D_ij(particle1,particle2)
        delta=self.delta_tensor(particle1,particle2)
        cm_pe_pe=self.Cm_pe_pe(particle1,D)
        cm_pa_pa=self.Cm_pa_pa(particle1,D)
        M=-fluid.rho*np.pi*particle1.diameter**3/6*((cm_pa_pa- cm_pe_pe)*delta+cm_pe_pe*I)
        return M

    def M_ind_matrix(self,particle1,particle2,fluid):
        """ Calculates the induced virtual mass matrix
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
            fluid: the fluid object.
        Returns:
           M: the induce virtual mass matrix
        """
        I=np.identity(3)
        D=self.D_ij(particle1,particle2)
        delta=self.delta_tensor(particle1,particle2)
        cm_ind_pe_pe=self.Cm_ind_pe_pe(particle1,D)
        cm_ind_pa_pa=self.Cm_ind_pa_pa(particle1,D)
        M_ind=-fluid.rho*np.pi*particle1.diameter**3/6*((cm_ind_pa_pa- cm_ind_pe_pe)*delta+cm_ind_pe_pe*I)
        return M_ind

    # forces function
    def F_added_mass(self,particle1,particle2,fluid):
        """ Calculates the  virtual mass force
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
            fluid: the fluid object.
        Returns:
            the virtual mass force on particle1.
        """
        acceleration1=(particle1.velocity-particle1.last_velocity)/self.dt
        acceleration2=(particle2.velocity-particle2.last_velocity)/self.dt

        m=self.M_matrix(particle1,particle2,fluid)
        m_ind=self.M_ind_matrix(particle1,particle2,fluid)    
        return np.matmul(m,acceleration1) +np.matmul(m_ind,acceleration2)

    def F_meshchersky(self,particle1,particle2,fluid):
        """ Calculates the  meshchersky force
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
            fluid: the fluid object.
        Returns:
            the virtual mass force on particle1.
        """
        m=self.M_matrix(particle1,particle2,fluid)
        m_ind=self.M_ind_matrix(particle1,particle2,fluid)
        dm_dt=(m-particle1.last_M)/self.dt
        dm_ind_dt=(m_ind-particle1.last_M_ind)/self.dt
        # store the matrix for the calculation in the next time step.
        particle1.last_M=m
        particle1.last_M_ind=m_ind
        return -fluid.rho*np.pi*particle1.diameter**3/6*(np.matmul(dm_dt,particle1.velocity) +np.matmul(dm_ind_dt,particle2.velocity) )

    def Cd_0(self,Re):
        """ Calculates the drag coefficient for a lone particle. Cliff et al. ISBN	0486445801, 9780486445809
        Args:
           Re: the Reynolds number
        Returns:
            Cd_0: the drag coefficient.
        """
        if Re<=20:
            return 24/Re*(1+0.1315*Re**(0.82-0.05*np.log10(Re)))
        else:
            return 24/Re*(1+0.1915*Re**(0.6305))    
        
    def F_drag(self,particle1,particle2,fluid1):
        """ Calculates the drag force
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the drag force.
        """
        velocity_for_calculation=-particle1.velocity
        if np.linalg.norm(particle1.velocity)<1e-10:
            return 0*particle1.velocity
        Re=np.linalg.norm(particle1.velocity)*particle1.diameter*fluid1.rho/fluid1.mu
        Cd0=self.Cd_0(Re+1e-30)
        d_position=(particle1.position-particle2.position)/particle1.diameter

        e=max(np.linalg.norm(d_position),1.0)
        e=min(e,8.0)
        to_acos=(np.dot(d_position,velocity_for_calculation)/(np.linalg.norm(velocity_for_calculation)+1e-30))/e
        if(to_acos>1):
            Theta=0
        elif (to_acos<-1):
            Theta=np.pi
        else:
            Theta=np.arccos((np.dot(d_position,velocity_for_calculation)/(np.linalg.norm(velocity_for_calculation)+1e-30))/e)

        x=np.array([[Re,e,Theta]])
        # evaluate the drag model
        x_input=self.scaler_x_cd.transform(x)
        NN_cd=self.model_cd.predict(x_input,verbose=0)
        Cd= self.scaler_y_cd.inverse_transform(NN_cd)
        Cd=Cd0*Cd[0][0]

        return -1/8*fluid1.rho*np.pi*particle1.diameter**2*Cd*particle1.velocity*np.linalg.norm(particle1.velocity)

    def F_lift(self,particle1,particle2,fluid):
        """ Calculates the lift force
        Args:
            particle1: the reference particle in the calculation. Usually the particle for which we are currently calculating the force. 
            particle2: the second particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the lift force.
        """
        velocity_for_calculation=-(particle1.velocity)
        if np.linalg.norm(particle1.velocity)<1e-10:
            return 0*particle1.velocity
        Re=np.linalg.norm(particle1.velocity)*particle1.diameter*fluid.rho/fluid.mu
        Cd0=self.Cd_0(Re+1e-30)
        d_position=(particle1.position-particle2.position)/particle1.diameter

        e=max(np.linalg.norm(d_position),1.0)
        to_acos=(np.dot(d_position,velocity_for_calculation)/(np.linalg.norm(velocity_for_calculation)+1e-30))/e
        if(to_acos>1):
            Theta=0
        elif (to_acos<-1):
            Theta=np.pi
        else:
            Theta=np.arccos((np.dot(d_position,velocity_for_calculation)/(np.linalg.norm(velocity_for_calculation)+1e-30))/e)
        # evaluate the lift model
        x=np.array([[Re,e,Theta]])
        x_input=self.scaler_x_cl.transform(x)
        NN_cl=self.model_cl.predict(x_input,verbose=0)
        Cl= self.scaler_y_cl.inverse_transform(NN_cl)
        Cl=Cl[0][0]*Cd0

        direction=d_position-np.dot(d_position,velocity_for_calculation)/(np.linalg.norm(velocity_for_calculation)**2+1e-30)*velocity_for_calculation
        direction=direction/(np.linalg.norm(direction)+1e-30)

        return 1*1/8*fluid.rho*np.pi*particle1.diameter**2*Cl*np.linalg.norm(particle1.velocity)**2*direction

    def F_buoyancy(self,particle1,fluid,g):
        """ Calculates the buoyancy force
        Args:
            particle1: the reference particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the lift buoyancy force.
        """
        return (particle1.rho-fluid.rho)*np.pi*particle1.diameter**3/6*g


    def F_lubrication(self,particle1,particle2,fluid):
        """ Calculates the lubrication force
        Args:
            particle1: the reference particle in the calculation.
            particle2: the second particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the lubrication force.
        """
        force_direction=(particle1.position-particle2.position)/np.linalg.norm(particle1.position-particle2.position)
        centroid_vector=(particle2.position-particle1.position)/np.linalg.norm((particle2.position-particle1.position))
        v_ij=np.dot(-particle1.velocity,force_direction)+np.dot(particle2.velocity,force_direction)
        kappa=particle2.diameter/particle1.diameter
        epsilone=(np.linalg.norm(particle1.position-particle2.position)-(particle1.diameter+particle2.diameter)/2)*2/particle1.diameter
        epsilone_ref=4.0
        if epsilone> epsilone_ref:
            return 0*force_direction
        elif epsilone<0.01:
            epsilone=0.01

        velocity_diff=particle2.velocity-particle1.velocity
        omega_diff=particle2.omega+particle1.omega
        normal_component=(kappa**2/(1+kappa)**2*(1/epsilone-1/epsilone_ref)-kappa*(1+7*kappa+kappa**2)/(5*(1+kappa)**3)*np.log(epsilone/epsilone_ref))*np.dot(velocity_diff,centroid_vector)*centroid_vector*6.0*np.pi*fluid.mu*(particle1.diameter/2)
        tangential_component_translation=1*-4*kappa*(2+kappa+2*kappa**2)/(15*(1+kappa)**3)*(velocity_diff-np.dot(velocity_diff,centroid_vector)*centroid_vector)*np.log(epsilone/epsilone_ref)*6.0*np.pi*fluid.mu*(particle1.diameter/2)
        tangential_component_rotation=1*2*kappa**2/(15*(1+kappa)**2)*np.cross((omega_diff+4*kappa**-1*particle1.omega+4*kappa*particle2.omega),centroid_vector)*np.log(epsilone/epsilone_ref)*6.0*np.pi*fluid.mu*(particle1.diameter/2)**2

        return normal_component+tangential_component_translation+tangential_component_rotation

    def F_magnus(self,particle1,fluid1):
        """ Calculates the magnus force.  Loth: https://doi.org/10.2514/1.29159
        Args:
            particle1: the reference particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the magnus force.
        """
        if np.linalg.norm(particle1.omega)>1e-10 and np.linalg.norm(particle1.velocity)>1e-10:
            force_direction=np.cross(-particle1.velocity,particle1.omega)
            
            Omega=particle1.diameter*np.linalg.norm(particle1.omega)/(2*np.linalg.norm(particle1.velocity)+1e-30)

            Re=np.linalg.norm(particle1.velocity)*particle1.diameter*fluid1.rho/fluid1.mu
            
            Cl=1.0-(0.675+0.15*(1.0+np.tanh(0.28*(Omega-2.0))))*np.tanh(0.18*np.sqrt(Re))

            return 1.0/8.0*fluid1.rho*np.pi*particle1.diameter**3*Cl*force_direction
        else:
            return 0*particle1.velocity # return a 0 vector

    def F_contact(self,particle1,particle2):
        """ Calculates the contact force with a soft sphere model
        Args:
            particle1: the reference particle in the calculation.
            particle2: the second particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the contact force.
        """
        gap=np.linalg.norm(particle1.position-particle2.position)-(particle1.diameter+particle2.diameter)/2
        d_position=(particle2.position-particle1.position)
        normal_vector=d_position/np.linalg.norm(d_position)
        if(gap>=0):
            particle1.tengential_overlap= 0*normal_vector
            particle1.is_in_contact=False
            return 0*normal_vector
        else:
            if(np.linalg.norm(0.5*particle1.diameter*particle1.omega+0.5*particle2.diameter*particle2.omega)>1e-10):
                v_ij=particle1.velocity-particle2.velocity+np.cross(0.5*particle1.diameter*particle1.omega+0.5*particle2.diameter*particle2.omega,normal_vector)
            else:
                v_ij=particle1.velocity-particle2.velocity

            if particle1.is_in_contact==False:
                particle1.contact_velocity=v_ij
                particle1.is_in_contact=True
            v_n=np.dot(v_ij,normal_vector)*normal_vector
            v_rt=v_ij-v_n
            particle1.tengential_overlap+=v_rt*self.dt
            re=(2.0/particle1.diameter+2.0/particle2.diameter)**-1
            me=(1.0/particle1.mass+1.0/particle2.mass)**-1
            ye=(1.0/particle1.young_modulus+1.0/particle2.young_modulus)**-1

            kn=16.0/15.0*np.sqrt(re)*ye*(15.0*me*np.linalg.norm(particle1.contact_velocity)**2/(16*np.sqrt(re)*ye))**0.2
            kt=0.4*kn
            cn=-2*np.log(particle1.restitution_coef)/np.sqrt(np.log(particle1.restitution_coef)**2+np.pi**2)*np.sqrt(me*kn)
            ct=-2*np.log(particle1.restitution_coef)/np.sqrt(np.log(particle1.restitution_coef)**2+np.pi**2)*np.sqrt(me*kt)

            normal_force=-(-kn*gap+cn*np.dot(v_ij,normal_vector))*normal_vector
            tangential_force=-(kt*particle1.tengential_overlap+ct*v_rt)
            if(np.linalg.norm(tangential_force)>np.linalg.norm(normal_force*particle1.friction_coef)):
                tangential_force=np.linalg.norm(normal_force*particle1.friction_coef)*tangential_force/np.linalg.norm(tangential_force)

            return normal_force+tangential_force
    
    def F_history(self,particle1,fluid1):
        """ Calculates the history force
        Args:
            particle1: the reference particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the history force.
        """
        # Kim et al 
        c1=2.5
        c2=0.126
        re_terminal=self.particle1.terminal_velocity*particle1.diameter*fluid1.rho/fluid1.mu
        acceleration1=(particle1.velocity-particle1.last_velocity)/self.dt
        Re=max(np.linalg.norm(particle1.velocity)*particle1.diameter*fluid1.rho/fluid1.mu,1e-6)
        particle_volume=particle1.diameter**3/6*np.pi
        Re_ref=re_terminal*0.001
        g_h_ref=(0.75+c2*Re_ref)/Re_ref
        mass_scale=9*particle_volume*fluid1.rho/(2*np.sqrt(np.pi))*(256.0/np.pi)**(1.0/6.0)*g_h_ref
        length_scale=particle1.diameter/2.0
        time_scale=particle1.diameter**2.0/(4.0*fluid1.mu/fluid1.rho)*(256.0/np.pi)**(1.0/3.0)*g_h_ref**2
        adimensional_time=self.t/time_scale
        adimensional_time_step=self.dt/time_scale
        addimential_accel=acceleration1*time_scale**2/length_scale
        mass_scale=9*particle_volume*fluid1.rho/(2*np.sqrt(np.pi))*(256.0/np.pi)**(1.0/6.0)*g_h_ref
        g_h=(0.75+c2*Re)/Re
        ri=(g_h_ref/g_h)**1.5
        gamma_i=ri**(1.0/3.0)*adimensional_time_step**0.25
        K0_i=2.0/9.0*ri**(-2.0/3.0)*(-0.3722*gamma_i+12.16*gamma_i**2-6.488*gamma_i**3)
        F_B_improper_near=-(K0_i*addimential_accel)*mass_scale*length_scale/time_scale**2
        F_B_near=0*acceleration1
        tau=0
        i=0
        while (tau+adimensional_time_step/2)<(adimensional_time-adimensional_time_step):
            if i==0:
                acceleration1_at_tau=(particle1.previous_velocity_list[i+1]-particle1.previous_velocity_list[i])/(self.dt)
                acceleration1_at_tau=acceleration1_at_tau*time_scale**2/length_scale
                Ki1=((adimensional_time-tau)**(0.5/c1)+ri*(adimensional_time-tau))**(-c1)
                acceleration2_at_tau=(particle1.previous_velocity_list[i+2]-particle1.previous_velocity_list[i])/(2*self.dt)
                acceleration2_at_tau=acceleration2_at_tau*time_scale**2/length_scale
                Ki2=((adimensional_time-(tau+adimensional_time_step))**(0.5/c1)+ri*(adimensional_time-(tau+adimensional_time_step)))**(-c1)
                F_B_near+=adimensional_time_step*(Ki1*acceleration1_at_tau+Ki2*acceleration2_at_tau)/2
            else:
                acceleration1_at_tau=(particle1.previous_velocity_list[i+1]-particle1.previous_velocity_list[i-1])/(2*self.dt)
                acceleration1_at_tau=acceleration1_at_tau*time_scale**2/length_scale
                Ki1=((adimensional_time-tau)**(0.5/c1)+ri*(adimensional_time-tau))**(-c1)
                acceleration2_at_tau=(particle1.previous_velocity_list[i+2]-particle1.previous_velocity_list[i])/(2*self.dt)
                acceleration2_at_tau=acceleration2_at_tau*time_scale**2/length_scale
                Ki2=((adimensional_time-(tau+adimensional_time_step))**(0.5/c1)+ri*(adimensional_time-(tau+adimensional_time_step)))**(-c1)
                F_B_near+=adimensional_time_step*(Ki1*acceleration1_at_tau+Ki2*acceleration2_at_tau)/2
            tau+=adimensional_time_step
            i+=1
        F_B_near=-F_B_near*(mass_scale)*(length_scale)/(time_scale)**2
        return (F_B_improper_near+ F_B_near)

    # Torque function
    def T_viscous_dissipation(self,particle1,fluid):
        """ Calculates the viscous torque
        Args:
            particle1: the reference particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the viscous torque.
        """
        re_omega=particle1.diameter**2*np.linalg.norm(particle1.omega)*fluid.rho/fluid.mu
        f_omega=1+5.0/64.0/np.pi*re_omega**0.6

        return -np.pi*particle1.diameter**3*particle1.omega*fluid.mu*f_omega
    
    def T_history(self,particle1,fluid):
        """ Calculates the torque due to the angular acceleration. 
        Args:
            particle1: the reference particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the torque.
        """
        d_omega=(particle1.omega-particle1.last_omega)/self.dt
        #### THIS TORQUE IS BEEN DISABLED ####
        #return -self.particle1.diameter**3/6*np.pi*self.fluid.rho*self.particle1.diameter**2/4*2*d_omega
        return 0*d_omega

    def T_induce(self,particle1,particle2,fluid):
        """ Calculate the induced torque . 
        Args:
            particle1: the reference particle in the calculation.
            particle2: the second particle in the calculation.
            fluid1: the fluid object.
        Returns:
            the torque.
        """
        velocity_for_calculation=-(particle1.velocity)
        if np.linalg.norm(particle1.velocity)<1e-10:
            return 0*particle1.velocity
        Re=np.linalg.norm(velocity_for_calculation)*particle1.diameter*fluid.rho/fluid.mu
        Cd0=self.Cd_0(Re+1e-30)
        d_position=(particle1.position-particle2.position)/particle1.diameter
        e=max(np.linalg.norm(d_position),1.0)
        to_acos=(np.dot(d_position,velocity_for_calculation)/(np.linalg.norm(velocity_for_calculation)+1e-30))/e
        if(to_acos>1):
            Theta=0
        elif (to_acos<-1):
            Theta=np.pi
        else:
            Theta=np.arccos((np.dot(d_position,velocity_for_calculation)/(np.linalg.norm(velocity_for_calculation)+1e-30))/e)
        # evaluate the torque model
        x=np.array([[Re,e,Theta]])
        x_input=self.scaler_x_ct.transform(x)
        NN_ct=self.model_ct.predict(x_input,verbose=0)
        Ct= self.scaler_y_ct.inverse_transform(NN_ct)
        Ct=Ct[0][0]*Cd0
        # define vector normal to particle plaine
        direction=np.cross(velocity_for_calculation,d_position)
        direction=direction/(np.linalg.norm(direction)+1e-30)
        return 1.0/16.0*fluid.rho*np.pi*particle1.diameter**3*Ct*np.linalg.norm(velocity_for_calculation)**2*direction

    def T_lubrication(self,particle1,particle2,fluid):
            """ Calculates the lubrication torque
            Args:
                particle1: the reference particle in the calculation.
                particle2: the second particle in the calculation.
                fluid1: the fluid object.
            Returns:
                the lubrication force.
            """
            force_direction=(particle1.position-particle2.position)/np.linalg.norm(particle1.position-particle2.position)
            centroid_vector=(particle2.position-particle1.position)/np.linalg.norm((particle2.position-particle1.position))
            v_ij=np.dot(-particle1.velocity,force_direction)+np.dot(particle2.velocity,force_direction)
            kappa=particle2.diameter/particle1.diameter
            epsilone=(np.linalg.norm(particle1.position-particle2.position)-(particle1.diameter+particle2.diameter)/2)*2/particle1.diameter

            epsilone_ref=4.0 # maximal gap for the lubrication force. (calibrated)
            if epsilone> epsilone_ref:
                return 0*force_direction
            elif epsilone<0.0625:
                epsilone=0.0625
            velocity_diff=particle2.velocity-particle1.velocity
            tangential_component_translation=-kappa**(4+kappa)/(10*(1+kappa)**2)*np.cross( centroid_vector,velocity_diff)*np.log(epsilone/epsilone_ref)*8.0*np.pi*fluid.mu*(particle1.diameter/2)**2
            tangential_component_rotation=2*kappa/(5*(1+kappa))*((particle1.omega+kappa*particle2.omega/4)-np.dot((particle1.omega+kappa*particle2.omega/4),centroid_vector)*centroid_vector)*np.log(epsilone/epsilone_ref)*8.0*np.pi*fluid.mu*(particle1.diameter/2)**3
            return tangential_component_translation+tangential_component_rotation
            
    def run(self):
        """ Run the model . 
        Returns:
            the output object with all the output variables
        """
        self.particle1.update()
        self.particle2.update()
        self.particle1.terminal_velocity=self.terminal_velocity(self.particle1)
        self.particle2.terminal_velocity=self.terminal_velocity(self.particle2)
        # Initialized the last virtual mass matrix with the initial position
        self.F_meshchersky(self.particle1,self.particle2,self.fluid)
        self.F_meshchersky(self.particle2,self.particle1,self.fluid) 
        t_total=np.zeros(int(np.ceil(self.final_time/self.dt))+1)
        i=1

        # Initialized output
        p1_position=[self.particle1.position]
        p2_position=[self.particle2.position]
        p1_velocity=[self.particle1.velocity]
        p2_velocity=[self.particle2.velocity]
        p1_omega=[self.particle1.omega]
        p2_omega=[self.particle2.omega]
        p1_total_fluid_force=[np.array([0.0,0.0,0.0])]
        p2_total_fluid_force=[np.array([0.0,0.0,0.0])]
        p1_total_fluid_torque=[np.array([0.0,0.0,0.0])]
        p2_total_fluid_torque=[np.array([0.0,0.0,0.0])]

        p1_virtual_mass_force=[np.array([0.0,0.0,0.0])]
        p2_virtual_mass_force=[np.array([0.0,0.0,0.0])]
        p1_meshchersky_force=[np.array([0.0,0.0,0.0])]
        p2_meshchersky_force=[np.array([0.0,0.0,0.0])]
        p1_drag_force=[np.array([0.0,0.0,0.0])]
        p2_drag_force=[np.array([0.0,0.0,0.0])]
        p1_lift_force=[np.array([0.0,0.0,0.0])]
        p2_lift_force=[np.array([0.0,0.0,0.0])]
        p1_lubrication_force=[np.array([0.0,0.0,0.0])]
        p2_lubrication_force=[np.array([0.0,0.0,0.0])]
        p1_magnus_force=[np.array([0.0,0.0,0.0])]
        p2_magnus_force=[np.array([0.0,0.0,0.0])]
        p1_history_force=[np.array([0.0,0.0,0.0])]
        p2_history_force=[np.array([0.0,0.0,0.0])]
        p1_contact_force=[np.array([0.0,0.0,0.0])]
        p2_contact_force=[np.array([0.0,0.0,0.0])]
        p1_buoyancy_force=[np.array([0.0,0.0,0.0])]
        p2_buoyancy_force=[np.array([0.0,0.0,0.0])]
        
        p1_induce_torque=[np.array([0.0,0.0,0.0])]
        p2_induce_torque=[np.array([0.0,0.0,0.0])]
        p1_viscous_torque=[np.array([0.0,0.0,0.0])]
        p2_viscous_torque=[np.array([0.0,0.0,0.0])]
        p1_history_torque=[np.array([0.0,0.0,0.0])]
        p2_history_torque=[np.array([0.0,0.0,0.0])]
        p1_contact_torque=[np.array([0.0,0.0,0.0])]
        p2_contact_torque=[np.array([0.0,0.0,0.0])]
        p1_lubrication_torque=[np.array([0.0,0.0,0.0])]
        p2_lubrication_torque=[np.array([0.0,0.0,0.0])]
      

        while self.t+ self.dt/2< self.final_time:
            print("current time = "+str(self.t))
            # evaluate the force and torque on particle 1
            self.particle1.previous_velocity_list.append(self.particle1.velocity)
            self.particle2.previous_velocity_list.append(self.particle2.velocity)
            self.particle1.previous_omega_list.append(np.copy(self.particle1.omega))
            self.particle2.previous_omega_list.append(np.copy(self.particle2.omega))   
            F_1_vm=self.F_added_mass(self.particle1,self.particle2,self.fluid)
            F_1_mesh=self.F_meshchersky(self.particle1,self.particle2,self.fluid)
            F_1_drag=self.F_drag(self.particle1,self.particle2,self.fluid)
            F_1_lift=self.F_lift(self.particle1,self.particle2,self.fluid)
            F_1_lub=self.F_lubrication(self.particle1,self.particle2,self.fluid)
            F_1_buoyancy=self.F_buoyancy(self.particle1,self.fluid,self.gravity)
            F_1_magnus=self.F_magnus(self.particle1,self.fluid)
            F_1_contact=self.F_contact(self.particle1,self.particle2)
            F_1_history=self.F_history(self.particle1,self.fluid)
            F_1=F_1_buoyancy+F_1_contact+F_1_drag+F_1_lift+F_1_lub+F_1_mesh+F_1_vm+F_1_magnus+ F_1_history
            T_1_dissipation=self.T_viscous_dissipation(self.particle1,self.fluid)
            T_1_induce=self.T_induce(self.particle1,self.particle2,self.fluid)
            T_1_history=self.T_history(self.particle1,self.fluid)
            T_1_lubrication=self.T_lubrication(self.particle1,self.particle2,self.fluid)
            if(np.linalg.norm(F_1_contact)>1e-10):
                T_1_contact=np.cross(F_1_contact,(self.particle2.position-self.particle1.position)*self.particle1.diameter/2.0/np.linalg.norm(self.particle2.position-self.particle1.position))
            else:
                T_1_contact=np.array([0.0,0.0,0.0])

            # evaluate the force and torque on particle 2
            F_2_vm=self.F_added_mass(self.particle2,self.particle1,self.fluid)
            F_2_mesh=self.F_meshchersky(self.particle2,self.particle1,self.fluid)
            F_2_drag=self.F_drag(self.particle2,self.particle1,self.fluid)
            F_2_lift=self.F_lift(self.particle2,self.particle1,self.fluid)
            F_2_lub=self.F_lubrication(self.particle2,self.particle1,self.fluid)
            F_2_buoyancy=self.F_buoyancy(self.particle2,self.fluid,self.gravity)
            F_2_magnus=self.F_magnus(self.particle2,self.fluid)
            F_2_contact=self.F_contact(self.particle2,self.particle1)
            F_2_history=self.F_history(self.particle2,self.fluid)
            F_2=F_2_buoyancy+F_2_contact+F_2_drag+F_2_lift+F_2_lub+F_2_mesh+F_2_vm+F_2_magnus+ F_2_history
            T_2_dissipation=self.T_viscous_dissipation(self.particle2,self.fluid)
            T_2_induce=self.T_induce(self.particle2,self.particle1,self.fluid)
            T_2_history=self.T_history(self.particle2,self.fluid)
            T_2_lubrication=self.T_lubrication(self.particle2,self.particle1,self.fluid)
            if(np.linalg.norm(F_2_contact)>1e-10):
                T_2_contact=np.cross(F_2_contact,(self.particle1.position-self.particle2.position)*self.particle2.diameter/2.0/np.linalg.norm(self.particle1.position-self.particle2.position))
            else:
                T_2_contact=np.array([0.0,0.0,0.0])

            # integrate 
            dv1=self.dt*F_1/self.particle1.mass
            dv2=self.dt*F_2/self.particle2.mass

            T_1=T_1_dissipation+T_1_induce+T_1_contact+ T_1_history+T_1_lubrication
            T_2=T_2_dissipation+T_2_induce+T_2_contact+T_2_history+T_2_lubrication
            do1=self.dt*T_1/self.particle1.inertia
            do2=self.dt*T_2/self.particle2.inertia

            self.particle1.omega=self.particle1.omega+do1
            self.particle2.omega=self.particle2.omega+do2      

            self.particle1.last_velocity=np.copy(self.particle1.velocity)
            self.particle2.last_velocity=np.copy(self.particle2.velocity)
            self.particle1.last_position=np.copy(self.particle1.position)
            self.particle2.last_position=np.copy(self.particle2.position)
            self.particle1.last_omega=np.copy(self.particle1.omega)
            self.particle2.last_omega=np.copy(self.particle2.omega)
            self.particle1.velocity=self.particle1.velocity+dv1
            self.particle2.velocity=self.particle2.velocity+dv2
            self.particle1.position=self.particle1.position+self.particle1.velocity*self.dt
            self.particle2.position=self.particle2.position+self.particle2.velocity*self.dt

            # store current results
            p1_position.append(self.particle1.position)
            p2_position.append(self.particle2.position)
            p1_velocity.append(self.particle1.velocity)
            p2_velocity.append(self.particle2.velocity)
            p1_omega.append(self.particle1.omega)
            p2_omega.append(self.particle2.omega)
            p1_total_fluid_force.append(F_1_drag+F_1_lift+F_1_lub+F_1_mesh+F_1_vm+F_1_magnus+ F_1_history)
            p2_total_fluid_force.append(F_2_drag+F_2_lift+F_2_lub+F_2_mesh+F_2_vm+F_2_magnus+ F_2_history)
            p1_total_fluid_torque.append(T_1_dissipation+T_1_induce+T_1_history)
            p2_total_fluid_torque.append(T_2_dissipation+T_2_induce+T_2_history)

            p1_virtual_mass_force.append(F_1_vm)
            p2_virtual_mass_force.append(F_2_vm)
            p1_meshchersky_force.append(F_1_mesh)
            p2_meshchersky_force.append(F_2_mesh)
            p1_drag_force.append(F_1_drag)
            p2_drag_force.append(F_2_drag)
            p1_lift_force.append(F_1_lift)
            p2_lift_force.append(F_2_lift)
            p1_lubrication_force.append(F_1_lub)
            p2_lubrication_force.append(F_2_lub)
            p1_magnus_force.append(F_1_magnus)
            p2_magnus_force.append(F_2_magnus)
            p1_history_force.append(F_1_history)
            p2_history_force.append(F_2_history)
            p1_contact_force.append(F_1_contact)
            p2_contact_force.append(F_2_contact)
            p1_buoyancy_force.append(F_1_buoyancy)
            p2_buoyancy_force.append(F_2_buoyancy)
            
            p1_induce_torque.append(T_1_induce)
            p2_induce_torque.append(T_2_induce)
            p1_viscous_torque.append(T_1_dissipation)
            p2_viscous_torque.append(T_2_dissipation)
            p1_history_torque.append(T_1_history)
            p2_history_torque.append(T_2_history)
            p1_contact_torque.append(T_1_contact)
            p2_contact_torque.append(T_2_contact)
            p1_lubrication_torque.append(T_1_lubrication)
            p2_lubrication_torque.append(T_2_lubrication)
      
            self.t+=self.dt
            t_total[i]=self.t
            i+=1
        
        #output
        return self.output(t_total,p1_position,p2_position,p1_velocity,p2_velocity,p1_omega,p2_omega,p1_total_fluid_force,p2_total_fluid_force,p1_total_fluid_torque,p2_total_fluid_torque,
                p1_virtual_mass_force,p2_virtual_mass_force,p1_meshchersky_force,p2_meshchersky_force,p1_drag_force,p2_drag_force,p1_lift_force,p2_lift_force,p1_lubrication_force,p2_lubrication_force,
                p1_magnus_force,p2_magnus_force,p1_history_force,p2_history_force,p1_contact_force,p2_contact_force,p1_buoyancy_force,p2_buoyancy_force,p1_induce_torque,
                p2_induce_torque,p1_viscous_torque,p2_viscous_torque,p1_history_torque,p2_history_torque,p1_contact_torque,p2_contact_torque,p1_lubrication_torque,p2_lubrication_torque)
