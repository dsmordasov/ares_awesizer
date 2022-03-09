"""This code was written in order to check the compliance of the final 
airborne wind energy system design against the system requirements.

Define the System, Environment and Cycle as `s, e, c` respectively.

Some example runs for e.g. optimisation purposes are shown at 
the end. Just press `F5` and make your CPU go brrrrrrrrrrrrrrrrrrrrrrrrrrrr
"""
#%% Imports, class definitions.
# =============================================================================
import numpy as np
from scipy import optimize as op
from scipy.integrate import quad
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import seaborn as sns
from style import set_graph_style
# =============================================================================


class Kite:
    """ Kite class.
    Attributes:
        c_l (float): Kite lift coefficient [-]
        c_d_k (float): Kite drag coefficient [-]
        flat_area (float):  Flattened kite area [m^2]
        flattening_factor (float): [-]
        projected_area (float): Effective kite aerodynamic area [m^2]
    """
    def __init__(self):
        self.c_l = 0.6
        self.c_d_k = 0.06
        #self.flat_area = 50.
        self.flattening_factor = 0.9
        self.area_density = 0.10802
        self.projected_area = 45.
        self.kite_mass = self.projected_area/self.flattening_factor*self.area_density
        self.kite_thickness = 0.01
        self.kite_volume = self.projected_area/self.flattening_factor*self.kite_thickness


    def calculate_kite_mass(self):
        self.kite_mass = self.projected_area/self.flattening_factor*self.area_density


class Tether:
    """ Tether class.
    Attributes:
        d_tether (float): Tether diameter [m]
        c_d_c (float): Cylinder drag coefficient [-]
        tether_max_length (float): Maximum tether length [m]
        tether_min_length (float): Minimum operational tether length [m]
        operational_length (float): Operational tether length [m]
        reeling_length (float): The length of tether that is reeled out during one \
            reel-out or reel-in phase, respectively [m]
    Methods:
        calculate_operational_length(): Calculates the operational tether length based on \
            Tether class attributes.
        calculate_reeling_length(): Calculates the reeling length for a single reel-out or reel-in \
            phase in a cycle
    """
    def __init__(self):
        self.nominal_tether_force = 7500
        self.c_d_c = 1.00
        self.tether_max_length = 400
        self.tether_min_length = 200
        self.design_factor = 3
        self.weight_slope = 2000
        self.GRAVITATIONAL_CONSTANT = 3.711
        self.mbl = (self.nominal_tether_force * self.design_factor)/self.GRAVITATIONAL_CONSTANT
        self.mass_tether_perhm = self.mbl/self.weight_slope
        self.tether_mass = self.mass_tether_perhm * self.tether_max_length / 100
        self.diameter_slope = 1.16/(np.pi*0.002**2)
        self.calculate_operational_length()
        self.calculate_reeling_length()
        self.calculate_tether_diameter()
        self.tether_volume = self.tether_max_length * self.d_tether**2 /4

    def calculate_operational_length(self):
        self.operational_length = (self.tether_max_length-self.tether_min_length)/2 +\
            self.tether_min_length
    
    def calculate_reeling_length(self):
        self.reeling_length = (self.tether_max_length - self.tether_min_length)
        
    def calculate_tether_diameter(self):
        self.max_break_load = self.nominal_tether_force*self.design_factor/\
            self.GRAVITATIONAL_CONSTANT
        self.tether_mass_per_hundred_m = self.max_break_load/self.weight_slope
        self.tether_mass = self.tether_max_length * self.tether_mass_per_hundred_m/100
        self.d_tether = 2 * np.sqrt(self.tether_mass_per_hundred_m/\
                                  (self.diameter_slope * np.pi))
    
    
class System(Kite, Tether):
    """ System class, inheriting all attributes and methods from Kite() and Tether().
    Attributes:
        reel_out_speed_limit (float): Maximum reel-out speed achievable [m/s]
        reel_in_speed_limit (float): Maximum reel-in speed achievable [m/s]
        cut_in_v_w (float): Cut-in wind speed [m/s]
        cut_out_v_w (float): Cut-out wind speed [m/s]
        nominal_tether_force (float): Maximum designed for tether force [N]
        nominal_generator_power (float): Maximum designed for generator power [W]
        phi (float): Elevation angle [rad]
        overall_gs_efficiency (float): Overall ground-station mechanical-to-electrical\
            conversion efficiency
        drum_outer_radius (float): Outer radius of the drum on which the tether is wound [m]
        drum_inner_radius (float): Inner radius of the drum on which the shaft is connect [m]
        drum_density (float): Material density of the drum (Aluminium 7075-T6) [kg / m^3]
        drum_length (float): Length of the drum [m]
        drum_inertia (float): Mass moment of inertia of the drum [kg / m^2]
        gen_speed_constant (float): Generator inverse of torque constant [A / N m]
        gen_terminal_resistance (float): Generator terminal resistance [Ohm]
        gen_other_losses (float): Factor accounting for higher resistance at operating\
            frequency due to the skin effect, stray-load-losses and other not explicitly\
                modelled losses of the generator
        gen_tau_s (float): Generator static contribution to friction torque [N m]
        gen_c_f (float): Generator dynamic contribution to friction torque [N s]
        mot_speed_constant (float): Motor inverse of torque constant [A / N m]
        mot_terminal_resistance (float): Motor terminal resistance [Ohm]
        mot_other_losses (float): Factor accounting for higher resistance at operating\
            frequency due to the skin effect, stray-load-losses and other not explicitly\
                modelled losses of the motor
        mot_tau_s (float): Motor static contribution to friction torque [N m]
        mot_c_f (float): Motor dynamic contribution to friction torque [N s]
        eta_energy_storage (float): Supercapacitor efficiency [-]
        eta_brake (float): Brakes efficiency [-]
        brake_power (float): Power required to release the brakes [W]
        spindle_power (float): Power required for the spindle motor [W]
        thermal_control_power (float): Power required for the heating system [J]   
        c_d (float): Drag coefficient of kite and tether [-]
        operational_height (float): Average height (z-coordinate) of operation [m]
    Methods:
        calculate_c_d(): Calculates drag coefficient of kite and tether based \
            on their inherited attributes
        calculate_operational_height(): Calculates average height \
            (z-coordinate) of operation 
    """
    def __init__(self):
        Kite.__init__(self)
        Tether.__init__(self)
        self.nominal_generator_power = 80000
        self.reel_out_speed_limit = 8
        self.reel_in_speed_limit = 25
        self.cut_in_v_w = 7
        self.cut_out_v_w = 35
        self.phi = 25.0 * np.pi/180.
        
        self.drum_outer_radius = 0.27
        self.drum_inner_radius = 0.26
        self.drum_length = (self.tether_max_length + 20) / (2 * np.pi * self.drum_outer_radius) * self.d_tether
        self.drum_volume = np.pi * (self.drum_outer_radius ** 2 - self.drum_inner_radius ** 2) * self.drum_length
        self.drum_density = 2810
        self.drum_mass = self.drum_volume * self.drum_density
        self.drum_inertia = 0.5 * self.drum_mass * (self.drum_outer_radius ** 2 + self.drum_inner_radius ** 2)
        
        self.gen_speed_constant = 1/12
        self.gen_terminal_resistance = 0.04*1.2
        self.gen_other_losses = 0.9
        self.gen_tau_s = 3.18
        self.gen_c_f = 0.799
        self.mot_speed_constant = 1/12 # 0.1898
        self.mot_terminal_resistance = 0.08*1.35
        self.mot_other_losses = 0.9
        self.mot_tau_s = 3.18
        self.mot_c_f = 0.799
        self.eta_energy_storage = 0.95
        self.eta_brake = 0.95
        self.spindle_power = 0
        self.thermal_control_power = 0
        
        self.calculate_c_d()
        self.calculate_operational_height()
        
        self.energy_requirement = np.genfromtxt("energy_req.csv", delimiter = ",")*1e3
        
    def calculate_c_d(self):
        self.c_d = self.c_d_k + \
            0.25*self.d_tether*self.operational_length*self.c_d_c*(1/self.projected_area)
    
    def calculate_operational_height(self):
        self.operational_height = np.sin(self.phi)*self.operational_length
        
class Environment(System):
    """ Environment class.
    Attributes:
        rho (float): Atmospheric density [kg/m^3]
        h_0 (float): Surface roughness [m]
        kappa (float): von Karman constant [-]
        friction_velocity (float): Friction velocity [m/s]
        k (float): Weibull k parameter [-]
        u (float): Weibull u parameter [m/s]
        v_w (float): Wind speed (at operational height) [m/s]
        season (int): Season counter, 0 = spring, 1 = summer, 2 = autumn, 3 = winter, 4 = dust storm
        dust_storm (bool): If True, then it considers a winter with a dust storm, of length dust_storm_length
        dust_storm_length (int): Length of the dust storm. Should range from 35 to 70 sols.
        g_w (float): Probability density for the given wind speed [-]
        ls_min_temp (array): List of solar longitudes values at minimum temperature [deg]
        ls_max_temp (array): List of solar longitudes values at maximum temperature [deg]
        rho_min_temp (array): List of the density values corresponding to the minimum temperatures [kg/m^3]
        rho_min_temp (array): List of the density values corresponding to the maximum temperatures [kg/m^3]
        ls_to_sol (array): List that compares sols to their respective solar longitudes
        sol_hours (array): List of day martian hours per sol [h]
    Methods:
        set_v_w(v_w): Updates the wind velocity
        set_friction_velocity(sol): Updates the environment friction velocity based on the sol
        set_weibull_k(sol): Updates the Weibull k parameter, based on the sol
        set_weibull_u(system): Updates the Weibull u parameter, based on System.operational_height
        calculate_weibull_pdf(v_w): Calculates the probability of v_w for the current Weibull distribution
        rho_interpolated_min_temp(sol): Interpolation function of the densities at minimum temperatures
        rho_interpolated_max_temp(sol): Interpolation function of the densities at maximum temperatures
        update_environment(system, sol): Updates all sol-dependent attributes of the environment
    """
    def __init__(self):
        self.rho = 0.021
        self.h_0 = 0.0316
        self.kappa = 0.4
        self.friction_velocity = 0.357
        self.k = 1.3
        self.v_w = 7
        self.season = 0
        self.dust_storm = True
        self.dust_storm_length = 35
        self.set_friction_velocity()
        self.set_weibull_k()
        self.set_weibull_u(System())
        
        self.ls_min_temp, self.rho_min_temp, self.ls_max_temp, self.rho_max_temp =\
            np.loadtxt("density.txt", unpack = True, delimiter = ',')
        self.ls_to_sol = np.loadtxt("solar_longitudes.txt", unpack = True)
        
        self.rho_interpolated_max_temp = scipy.interpolate.interp1d(self.ls_max_temp, self.rho_max_temp) 
        self.rho_interpolated_min_temp = scipy.interpolate.interp1d(self.ls_min_temp, self.rho_min_temp) 
        
        self.sol_hours = np.genfromtxt('day_length.csv', delimiter=",")
        for i in range(0,len(self.sol_hours)):
            if i >= 0 and i <= 514:
                self.sol_hours[i] += 2 # An hour of operation before dawn and one after dusk added.
            elif i > 514:
                self.sol_hours[i] += 4 # Winter 24/7 operation.
        self.sols = np.arange(0,len(self.sol_hours),1)    
        
    def set_v_w(self, v_w):
        self.v_w = v_w
        
    def set_season(self, sol):
        if sol >= 0 and sol <= 194: # Spring
            self.season = 0
        elif sol > 194 and sol <= 372: # Summer
            self.season = 1 
        elif sol > 372 and sol <= 514: # Autumn
            self.season = 2
        elif (sol > 514 and sol < (514 + self.dust_storm_length)) and self.dust_storm: # Winter w/ dust storm
            self.season = 4
        elif sol > 514 and sol <= 669: # Winter
            self.season = 3
    
    def set_friction_velocity(self):
        if self.season == 0: # Spring
            friction_velocity = 0.357 
        elif self.season == 1: # Summer
            friction_velocity = 0.265  
        elif self.season == 2: # Autumn
            friction_velocity = 0.459
        elif self.season == 3: # Winter
            friction_velocity = 0.764
        elif self.season == 4:
            friction_velocity = 0.662
        self.friction_velocity = friction_velocity
    
    def set_weibull_k(self):
        if self.season == 0: # Spring
            k = 1.3 
        elif self.season == 1: # Summer
            k = 1.58  
        elif self.season == 2: # Autumn
            k = 1.28
        elif self.season == 3: # Winter
            k = 1.6
        elif self.season == 4: # Dust storm
            k = 1.3
        self.k = k
    
    def set_weibull_u(self, system):
        self.u = (self.friction_velocity/self.kappa) *\
            np.log(system.operational_height/self.h_0)
            
    def calculate_weibull_pdf(self, v_w):
        self.g_w = (self.k/self.u)*\
            ((v_w/self.u)**(self.k-1))*\
                np.exp(-(self.v_w/self.u)**self.k)
    
    def set_density(self, sol):
        """
        Attributes:
            rho_low (float): The low value of the atmospheric density at a certain sol [kg/m^3]
            rho_high (float): The high value of the atmospheric density at a certain sol [kg/m^3]
            rho_avg (float): The average value of the atmospheric density at a certain sol [kg/m^3]
        """
        self.ls = self.ls_to_sol[sol]
        self.rho_low = float(self.rho_interpolated_max_temp(self.ls))*1e-3 
        self.rho_high = float(self.rho_interpolated_min_temp(self.ls))*1e-3 
        self.rho_avg = (self.rho_low + self.rho_high)/2
        self.rho = self.rho_low
    
    def update_environment(self, system, sol):
        self.set_season(sol)
        self.set_density(sol)
        self.set_friction_velocity()
        self.set_weibull_k()
        self.set_weibull_u(system)


class Cycle():
    """Cycle class, for running the analysis.
    Attributes:
        region_counter (integer): 1, 2 (limit tether force) or 3 (limit power), depending on the operational region [-]
        setting (integer): 2 or 3, depending on whether a 2-phase or 3-phase strategy is used [-]
    Methods:
        calculate_generator_eta(system): Calculates generator efficiency
            Attributes:
                tau_f_out (float): Friction torque 
        run_simulation(System, Environment, setting = 2): Finds the kite state for reel-out \
            and reel-in phase for current System and Environment\
            Attributes:
                f_out (float): Dimensionless force factor for reel-out phase [-]
                f_in (float): Dimensionless force factor for reel-in phase [-]
                gamma_out_max (float): Maximum dimensionless reel-out velocity factor [-]
                gamma_in_max (float): Maximum dimensionless reel-in velocity factor [-]
                gamma_out (float): Dimensionless reel-out velocity factor [-]
                gamma_in (float): Dimensionless reel-in velocity factor [-]
                v_out (float): Reel-out speed [m/s]
                v_in (float): Reel-in speed [m/s]
                tether_force_out (float): Tether traction force [N]
                tether_force_in (float): Tether retraction force [N]
                torque_out (float): Mechanical torque for the reel-out phase [N m]
                torque_in (float): Mechanical torque for the reel-in phase [N m]
                rpm_out (float): Rotational speed at reel-out [rpm]
                rpm_in (float): Rotational speed at reel-in [rpm]
                rpm_out_max (float): Maxmum rotational speed at reel out [rpm]
                rpm_in_max (float): Maxmum rotational speed at reel in [rpm]
                cycle_energy (float): Energy produced over the whole cycle [J]
                cycle_out_time (float): Time taken for the reel-out phase [s]
                cycle_in_time (float): Time taken for the reel-in phase [s]
                cycle_time (float): Time taken for the whole cycle [s]
                cycle_power (float): Power produced during the whole cycle [W]
                cycle_out_power (float): Power that is produced during the reel-out phase [W]
                cycle_in_power (float): Power used to complete the reel-in phase [W]
                nominal_v_w (float): Wind speed at which the first limit is reached \
                    force limits are reached [m/s]
                nominal_gamma_out (float): Optimised dimensionless reel-out velocity \
                    parameter for the moment nominal_v_w was encountered.
                mu (float): Dimensionless velocity parameter [-]
                f_out_mu (float): Dimensionless force factor for reel-out phase, region 3 [-]      
    """
    def __init__(self):
        self.region_counter = 1
        self.setting = 3
        self.cycle_max_time = 0
    
    def calculate_generator_eta(self, system):
        #self.gen_eta = 1
        self.tau_f_out = system.gen_tau_s + (system.gen_c_f * system.drum_outer_radius * self.rpm_out * 2 * np.pi)/60
        self.tau_g = self.torque_out - self.tau_f_out
        self.i_out = self.tau_g * system.gen_speed_constant
        self.le_out = 3 * system.gen_terminal_resistance * self.i_out**2 / system.gen_other_losses
        self.pm_out = self.tau_g * self.rpm_out * 2 * np.pi / 60
        self.pe_out = self.pm_out - self.le_out
        self.gen_eta = self.pe_out / self.pm_out

    def calculate_motor_eta(self, system):
        #self.mot_eta = 1
        self.tau_f_in = system.mot_tau_s + (system.mot_c_f * system.drum_outer_radius * self.rpm_in * 2 * np.pi)/60
        self.tau_m = self.torque_in + self.tau_f_in
        self.i_in = self.tau_m * system.mot_speed_constant
        self.le_in = 3 * system.mot_terminal_resistance * self.i_in**2 / system.mot_other_losses
        self.pm_in = self.tau_m * self.rpm_in * 2 * np.pi / 60
        self.pe_in = self.pm_in + self.le_in
        self.mot_eta = self.pm_in / self.pe_in
    
    def run_simulation(self, system, environment):
        self.gamma_out_max = system.reel_out_speed_limit/environment.v_w
        self.gamma_in_max = system.reel_in_speed_limit/environment.v_w
         
        self.rpm_out_max = system.reel_out_speed_limit / system.drum_outer_radius * 60 / 2 / np.pi
        self.rpm_in_max = system.reel_in_speed_limit / system.drum_outer_radius * 60 / 2 / np.pi
        
        system.brake_energy = 0.5 * system.drum_inertia * max(self.rpm_out_max * 2 * np.pi / 60, \
                                        self.rpm_in_max * 2 * np.pi / 60) ** 2 / system.eta_brake
        
        if (self.region_counter == 1):
            self.f_out = (system.c_l**3)/(system.c_d**2)
            self.f_in = system.c_d
            
            def objective_function(x):
                gamma_out = x[0]
                gamma_in = x[1]
                f_c = ((1-gamma_out)**2 - (self.f_in/self.f_out)*(1+gamma_in)**2)*\
                    ((gamma_out*gamma_in)/(gamma_out+gamma_in))
                return -f_c
            
            starting_point = (0.001, 0.001)
            bounds = ((0.001, self.gamma_out_max),
                      (0.001, self.gamma_in_max),)
            
            optimisation_result = op.minimize(objective_function, starting_point,
                                              bounds=bounds, method='SLSQP')
            self.gamma_out = optimisation_result['x'][0]
            self.gamma_in = optimisation_result['x'][1]
            
            self.v_out = environment.v_w*self.gamma_out
            self.v_in = environment.v_w*self.gamma_in
            
            self.tether_force_out = 0.5*environment.rho*(environment.v_w**2)*\
                system.projected_area*((1-self.gamma_out)**2)*self.f_out  
            self.torque_out = self.tether_force_out * system.drum_outer_radius
            self.rpm_out = (self.v_out / system.drum_outer_radius) * 60 / (2*np.pi)
            
            self.tether_force_in = 0.5*environment.rho*(environment.v_w**2)*\
                system.projected_area*((1+self.gamma_in)**2)*self.f_in
            self.torque_in = self.tether_force_in * system.drum_outer_radius
            self.rpm_in = (self.v_in / system.drum_outer_radius) * 60 / (2*np.pi)
            
            if ((self.setting == 2) and (self.tether_force_out >= system.nominal_tether_force or self.tether_force_in > system.nominal_tether_force)):
                self.region_counter = 3
                self.nominal_v_w = environment.v_w
# =============================================================================
#                 print("Region 3 entered from Region 1 at {:.1f} m/s (tether force limitation)."\
#                       .format(environment.v_w))
# =============================================================================
            elif ((self.setting == 3) and (self.tether_force_out >= system.nominal_tether_force or self.tether_force_in > system.nominal_tether_force)):
                self.region_counter = 2
                self.nominal_v_w = environment.v_w
# =============================================================================
#                 print("Region 2 entered from Region 1 at {:.1f} m/s (tether force limitation)."\
#                       .format(environment.v_w))
# =============================================================================
                self.nominal_gamma_out = self.gamma_out
 
            self.cycle_out_time = system.reeling_length/self.v_out
            self.cycle_in_time = system.reeling_length/self.v_in
            self.cycle_time = self.cycle_out_time + self.cycle_in_time
            
            self.calculate_generator_eta(system)
            self.cycle_out_power = self.pm_out * self.gen_eta
            self.calculate_motor_eta(system)
            self.cycle_in_power = (self.pm_in / self.mot_eta + system.spindle_power + \
                                   system.thermal_control_power) / system.eta_energy_storage
            
            self.cycle_power = (self.cycle_out_power*self.cycle_out_time - self.cycle_in_power*self.cycle_in_time - \
                                system.brake_energy / system.eta_energy_storage) / self.cycle_time

            if self.setting == 2 and self.cycle_out_power >= system.nominal_generator_power:
                self.region_counter = 3
                self.nominal_v_w = environment.v_w
# =============================================================================
#                 print("Region 3 entered from Region 1 at {:.1f} m/s (power limitation)."\
#                       .format(environment.v_w))
# =============================================================================
            elif self.setting == 3 and self.cycle_out_power >= system.nominal_generator_power:
                self.region_counter = 3
                self.nominal_v_w = environment.v_w
# =============================================================================
#                 print("Region 3 entered from Region 1 at {:.1f} m/s (power limitation)."\
#                       .format(environment.v_w))
# =============================================================================
                    
        if (self.region_counter == 2):
            self.tether_force_out = system.nominal_tether_force
            self.mu = environment.v_w / self.nominal_v_w
            self.gamma_out = 1 - ((1 - self.nominal_gamma_out) / self.mu)
            self.v_out = environment.v_w - self.nominal_v_w + self.nominal_gamma_out*self.nominal_v_w
            
            self.torque_out = self.tether_force_out * system.drum_outer_radius
            self.rpm_out = (self.v_out / system.drum_outer_radius) * 60 / (2*np.pi)
            
            self.calculate_generator_eta(system)
            self.cycle_out_power = self.pm_out * self.gen_eta
            # self.cycle_out_power = self.tether_force_out*self.v_out*self.gen_eta
            
            if self.cycle_out_power >= system.nominal_generator_power:
                self.region_counter = 3
# =============================================================================
#                 print("Region 3 entered from Region 2 at {:.1f} m/s (power limitation)."\
#                       .format(environment.v_w))
# =============================================================================
                self.nominal_v_w = environment.v_w
            
            def objective_function(x):
                gamma_in = x[0]
                f_c_2 = ((1/(self.mu**2))*(1-self.nominal_gamma_out)**2 - (self.f_in/self.f_out)*(1+gamma_in)**2)*\
                    ((gamma_in*(self.mu - 1 + self.nominal_gamma_out))/(self.mu*gamma_in + self.mu - 1 +\
                                                                        self.nominal_gamma_out))
                return -f_c_2
            
            starting_point = (0.001)
            bounds = ((0.001, self.gamma_in_max),)
            
            optimisation_result = op.minimize(objective_function, starting_point, bounds=bounds, method='SLSQP')
            self.gamma_in = optimisation_result['x'][0]
            
            self.v_in = environment.v_w * self.gamma_in
            self.tether_force_in = 0.5*environment.rho*(environment.v_w**2)*\
                system.projected_area*((1+self.gamma_in)**2)*self.f_in
                
            self.torque_in = self.tether_force_in * system.drum_outer_radius
            self.rpm_in = (self.v_in / system.drum_outer_radius) * 60 / (2*np.pi)
            
            self.calculate_motor_eta(system)
            self.cycle_in_power = (self.pm_in / self.mot_eta + system.spindle_power + \
                                  system.thermal_control_power) / system.eta_energy_storage
                
            self.cycle_out_time = system.reeling_length/self.v_out
            self.cycle_in_time = system.reeling_length/self.v_in
            self.cycle_time = self.cycle_out_time + self.cycle_in_time
            
            self.cycle_power = (self.cycle_out_power*self.cycle_out_time - self.cycle_in_power*self.cycle_in_time - \
                                system.brake_energy / system.eta_energy_storage) / self.cycle_time
            
            
        if (self.region_counter == 3):
            self.tether_force_out = system.nominal_tether_force
            self.cycle_out_power = system.nominal_generator_power
            
            
            self.v_out = self.cycle_out_power / (self.tether_force_out * self.gen_eta)
            
            self.torque_out = self.tether_force_out * system.drum_outer_radius
            self.rpm_out = (self.v_out / system.drum_outer_radius) * 60 / (2*np.pi)
            self.calculate_generator_eta(system)
            
            self.gamma_out = self.v_out / environment.v_w
            self.mu = environment.v_w / self.nominal_v_w
            
            def objective_function(x):
                gamma_in = x[0]
                f_c_3 = ((1/(self.mu**2))*(1-self.gamma_out)**2 - (self.f_in/self.f_out)*(1+gamma_in)**2)*\
                    ((self.gamma_out*gamma_in)/(self.gamma_out+self.mu*gamma_in))
                return -f_c_3
            
            starting_point = (0.001)
            bounds = ((0.001, self.gamma_in_max),)
            
            optimisation_result = op.minimize(objective_function, starting_point, bounds=bounds, method='SLSQP')
            self.gamma_in = optimisation_result['x'][0]
            
            self.v_in = environment.v_w * self.gamma_in
            self.tether_force_in = 0.5*environment.rho*(environment.v_w**2)*\
                system.projected_area*((1+self.gamma_in)**2)*self.f_in
            self.torque_in = self.tether_force_in * system.drum_outer_radius
            self.rpm_in = (self.v_in / system.drum_outer_radius) * 60 / (2*np.pi)
            self.calculate_motor_eta(system)
            self.cycle_in_power = (self.pm_in / self.mot_eta + system.spindle_power + \
                                   system.thermal_control_power) / system.eta_energy_storage
            
            self.cycle_out_time = system.reeling_length/self.v_out
            self.cycle_in_time = system.reeling_length/self.v_in
            self.cycle_time = self.cycle_out_time + self.cycle_in_time
            
            self.cycle_power = (self.cycle_out_power*self.cycle_out_time - self.cycle_in_power*self.cycle_in_time - \
                                system.brake_energy / system.eta_energy_storage) / self.cycle_time
            
            if self.cycle_in_time > self.cycle_max_time:
                self.cycle_max_time = self.cycle_in_time


def annual_calculation(s, e, graph=False):
    """Calculate the daily wind energy produced for the whole year.
    Also makes two .csv files and a pretty sexy graph.
    """
    start_time = time.time()
    
    energy_mwh = np.zeros(669)
    
    def average_power(v_w):
        e.set_v_w(v_w)
        e.calculate_weibull_pdf(v_w)
        c.run_simulation(s, e)
        return c.cycle_power*e.g_w
    
    for sol in e.sols:
        e.update_environment(s, sol)
        integrated_power = quad(average_power, s.cut_in_v_w, s.cut_out_v_w, limit=10, epsabs=1., epsrel=1.)
        energy_mwh[sol] = integrated_power[0]*e.sol_hours[sol]*1e-6
    
    energy_mwh *= np.cos(s.phi)**3
    np.savetxt("wind_energy_produced.csv", energy_mwh, delimiter=",")
    
    energy_mwh_req = s.energy_requirement*1e-6
    requirement_fulfilment = energy_mwh / energy_mwh_req
    np.savetxt("wind_energy_fractions.csv", requirement_fulfilment, delimiter=",")
            
    end_time = time.time()
    print("Runtime of {:.1f} seconds.".format(end_time-start_time))
    print("System produced {:.1f} MWh over the year.".format(np.sum(energy_mwh)))
    print("That is a difference of {:.1f} MWh w.r.t. the required {:.1f} MWh."\
          .format(np.abs(np.sum(energy_mwh_req)-np.sum(energy_mwh)),np.sum(energy_mwh_req)))
    
    test_result_text = "The energy production requirements are fulfiled."
    for sol in e.sols:
        if (requirement_fulfilment[sol] != 0 and requirement_fulfilment[sol] < 1):
            test_result_text = "The energy production requirements are NOT fulfilled (sol {}, {:.2f})."\
                .format(sol, requirement_fulfilment[sol])
            break
    print(test_result_text)
    
    if graph:
        set_graph_style()
        fig, ax = plt.subplots(2, 1, sharex=True)
        sns.lineplot(x = e.sols, y = requirement_fulfilment, color="darkblue", ax=ax[0])
        ax[0].set(title = "Daily wind energy production per year", ylabel = "Produced / required [-]")
        sns.lineplot(x = e.sols, y = energy_mwh, color = "darkblue", ax=ax[1])
        ax[1].set(xlabel = "Sol", ylabel = "Produced [MWh]")
        plt.subplots_adjust(bottom=0.1)
    
    return energy_mwh


def get_operational_envelope(plot=False):
    """Calculates the operational envelope of the generator/motor.
    If plot == True, this function also plots the operational envelope as 
    a function of wind speed.
    """
    wind_range = np.linspace(s.cut_in_v_w, s.cut_out_v_w, 35)
    reel_in_speeds = np.zeros(len(wind_range))
    reel_out_speeds = np.zeros(len(wind_range))
    tether_forces_in = np.zeros(len(wind_range))
    tether_forces_out = np.zeros(len(wind_range))
    taus_g = np.zeros(len(wind_range))
    taus_m = np.zeros(len(wind_range))
    
    for i in range(0,len(wind_range)):
        v_w = wind_range[i]
        c.run_simulation(s, e)
        reel_in_speeds[i] = c.v_in
        reel_out_speeds[i] = c.v_out
        tether_forces_out[i] = c.tether_force_out
        tether_forces_in[i] = c.tether_force_in
        taus_g[i] = c.tau_g
        taus_m[i] = c.tau_m
    
    rpms_in = (reel_in_speeds / s.drum_outer_radius) * 60 / (2 * np.pi)
    rpms_out = (reel_out_speeds / s.drum_outer_radius) * 60 / (2 * np.pi)
    torques_in = taus_m
    torques_out = taus_g
    
    # Calculate operational range for the generator
    gen_rpm_max = int(max(rpms_out))
    gen_rpm_min = int(min(rpms_out))
    
    gen_torque_max = int(max(torques_out))
    gen_torque_min = int(min(torques_out))
    
    
    # Calculate operational range for the motor
    mot_rpm_max = int(max(rpms_in))
    mot_rpm_min = int(min(rpms_in))
    
    mot_torque_max = int(max(torques_in))
    mot_torque_min = int(min(torques_in))
    
    if plot == True:
        # Print operational range
        print(f'The generator must operate at a rotational speed between {gen_rpm_max}-{gen_rpm_min} RPM and a torque between {gen_torque_max}-{gen_torque_min} Nm.\n')
        print(f'The motor must operate at a rotational speed between {mot_rpm_max}-{mot_rpm_min} RPM and a torque between {mot_torque_max}-{mot_torque_min} Nm.')
        
        # Plot reeling speeds and (re-)traction force as a function of wind speed
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(wind_range, reel_in_speeds, label="in")
        ax[0].plot(wind_range, reel_out_speeds, label="out")
        ax[0].legend()
        ax[1].plot(wind_range, tether_forces_in, label="in")
        ax[1].plot(wind_range, tether_forces_out, label="out")
        ax[1].legend()
        
        ax[0].set_ylabel('Reel speed [m/s]')
        ax[1].set_ylabel('Force [N]')
        ax[1].set_xlabel('Wind speed [m/s]')


        # Plot rotational speeds and torques as a function of wind speed
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(wind_range, rpms_in, label="in")
        ax[0].plot(wind_range, rpms_out, label="out")
        ax[0].legend()
        ax[1].plot(wind_range, torques_in, label="in", )
        ax[1].plot(wind_range, torques_out, label="out")
        ax[1].legend()
        
        ax[0].set_ylabel('Rotational speed [RPM]')
        ax[1].set_ylabel('Torque [Nm]')
        ax[1].set_xlabel('Wind speed [m/s]')
    
    return gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max


def get_gs_mass(gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max):
    """Function definition to estimate the mass of the ground station based on
    the Alxion catalogues (alternators & direct drive).  Returns the total 
    mass of the ground station. Brakes and spindle are fixed values that do 
    not scale depending on the operational envelope.
    """
    # Load technical information of generators and motors from Alxion catalogues
    generators = []
    motors = []
    
    with open('generators.csv') as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            line = [int(value) for value in line[:4]] + [float(line[-2])] + [line[-1]]
            generators.append(line)
    
    with open('motors.csv') as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            line = [int(round(float(value))) for value in line[:4]] + [float(line[-2])] + [line[-1]]
            motors.append(line)
    
    # Calculate mass of the generator part of the actuator
    generator_mass = 0
    
    if generator_mass == 0:
        for generator in generators:
            torque_range = range(int(generator[0]), int(generator[1]))
            rpm_range = range(int(generator[2]), int(generator[3]))
            
            if gen_torque_max in torque_range and gen_rpm_max in rpm_range and generator_mass == 0:
                generator_mass = generator[-2]
                print(f'The generator mass is estimated to be {round(generator_mass, 1)} kg.')
                print(f'The estimated generator mass corresponds to model {generator[-1]}.' )
              
        for i in range(len(generators) - 1):
            torque_range = range(int(generators[i + 1][1]), int(generators[i][1]))
            rpm_range = range(int(generators[i + 1][2]), int(generators[i][3]))
            
            if gen_torque_max in torque_range and gen_rpm_max in rpm_range and generator_mass == 0:
                calculate_gen_mass = lambda tau: np.interp(tau, [generators[i + 1][1], \
                          generators[i][1]], [generators[i + 1][-2], generators[i][-2]])
                generator_mass = calculate_gen_mass(gen_torque_max)
                
                print(f'The generator mass is estimated to be {round(generator_mass, 1)} kg.')
                print(f'The estimated generator mass is interpolated from models {generators[i + 1][-1]} and {generators[i][-1]}.\n')
   
    # Calculate mass of the motor part of the actuator
    motor_mass = 0
    
    if motor_mass == 0:
        for motor in motors:
            torque_range = range(int(motor[0]), int(motor[1]))
            rpm_range = range(int(motor[2]), int(motor[3]))
            
            if mot_torque_max in torque_range and mot_rpm_max in rpm_range and motor_mass == 0:
                motor_mass = motor[-2]
                print(f'The motor mass is estimated to be {round(motor_mass, 1)} kg.')
                print(f'The estimated motor mass corresponds to model {motor[-1]}.' )
              
        for i in range(len(motors) - 1):
            torque_range = range(int(motors[i + 1][1]), int(motors[i][1]))
            rpm_range = range(int(motors[i + 1][2]), int(motors[i][3]))
            
            if mot_torque_max in torque_range and mot_rpm_max in rpm_range and motor_mass == 0:
                calculate_mot_mass = lambda tau: np.interp(tau, [motors[i + 1][1], \
                          motors[i][1]], [motors[i + 1][-2], motors[i][-2]])
                motor_mass = calculate_mot_mass(gen_torque_max)
                
                print(f'The motor mass is estimated to be {round(motor_mass, 1)} kg.')
                print(f'The estimated motor mass is interpolated from models {motors[i + 1][-1]} and {motors[i][-1]}.')         
                
    
    # If you want this code to calculate the mass of a ground station with a ludicrous operational envelope, it won't like it
    if generator_mass == 0 or motor_mass == 0:
        raise Exception('The operational envelope of the ground station is unrealistic.')
    
    # Specify masses of other elements in the ground station
    actuator_mass = generator_mass + motor_mass
    spindle_mass = 5.20 + 20
    drum_mass = s.drum_mass
    brake_mass = 3
    print("generator: {:.2f}, motor: {:.2f}, drum: {:.2f}".format(generator_mass,\
                                                                  motor_mass, drum_mass))
    return actuator_mass + spindle_mass + drum_mass + brake_mass
#%% Main run
"""
Example of a main run. Gives you the system's AEP, compares it to the
AEP requirement of the system, warns you if the system underperforms for the
overall habitat requirements on a certain day (i.e. not compensated enough by
the solar panels and energy storage system). 
"""
s = System()
e = Environment()
c = Cycle()
energy_mwh = annual_calculation(s, e, graph=True)
gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max = get_operational_envelope(plot=False)
gs_mass = get_gs_mass(gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max)
system_mass = s.kite_mass + s.tether_mass + gs_mass + 5.3
print("System mass is {:.2f} kg.".format(system_mass))

#%% Optimisation run - choose attribute to check AEP for.

opt_attribute = "projected_area" # Change this line to choose attribute to be sized
attribute_array = np.array([45]) # Change this line to choose tried attribute values
iterations = range(0, len(attribute_array))
opt_results = np.zeros(len(attribute_array))

arr_gen_torque_max = np.zeros(len(attribute_array))
arr_gen_rpm_max = np.zeros(len(attribute_array))
arr_mot_torque_max = np.zeros(len(attribute_array))
arr_mot_rpm_max = np.zeros(len(attribute_array))
arr_gs_mass = np.zeros(len(attribute_array))
arr_system_mass = np.zeros(len(attribute_array))

for i in iterations:
    s = System()
    setattr(s, opt_attribute, attribute_array[i])
    s.calculate_kite_mass()
    e = Environment()
    c = Cycle()
    opt_results[i] = np.sum(annual_calculation(s, e))
    arr_gen_torque_max[i], arr_gen_rpm_max[i], arr_mot_torque_max[i], arr_mot_rpm_max[i] = get_operational_envelope(plot=False)
    arr_gs_mass[i] = get_gs_mass(arr_gen_torque_max[i], arr_gen_rpm_max[i], arr_mot_torque_max[i], arr_mot_rpm_max[i])
    arr_system_mass[i] = s.kite_mass + s.tether_mass + arr_gs_mass[i] + 5.3
    print("System mass is {:.2f} kg.".format(arr_system_mass[i]))
    
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(attribute_array, opt_results, 'o')
ax[1].plot(attribute_array, arr_system_mass, 'o')

ax[0].set_ylabel('AEP [MWh]')
ax[1].set_ylabel('Mass [kg]')
ax[1].set_xlabel(opt_attribute)
#%% Test run for one state.

s = System()
e = Environment()

c = Cycle()

e.season = 3
e.v_w = 25
c.run_simulation(s, e)
cap_en = c.cycle_in_power * (c.cycle_in_time/(60*60))
print("Power in: {:.1f} W for {:.1f} s, equalling {:.1f} Wh".format\
      (c.cycle_in_power, c.cycle_in_time, cap_en))

#%% Optimisation run - choose attribute to check AEP for.

opt_attribute = "nominal_tether_force" # Change this line to choose attribute to be sized
attribute_array = np.array([8000, 10000, 12000, 14000]) # Change this line to choose tried attribute values
iterations = range(0, len(attribute_array))
opt_results = np.zeros(len(attribute_array))
for i in iterations:
    s = System()
    setattr(s, opt_attribute, attribute_array[i])
    e = Environment()
    e.dust_storm = True
    e.dust_storm_length = 70
    c = Cycle()
    opt_results[i] = np.sum(annual_calculation(s, e))
plt.figure()
plt.plot(attribute_array,opt_results)
plt.xlabel(opt_attribute)
plt.ylabel("AEP [MWh]")

#%% Test run - graph power, force, wind speed
s = System()
e = Environment()
e.rho = 0.021

c = Cycle()

wind_range = np.linspace(7,30,35)

mean_powers = np.zeros(len(wind_range))
powers_out = np.zeros(len(wind_range))
powers_in = np.zeros(len(wind_range))
tether_forces_out = np.zeros(len(wind_range))
tether_forces_in = np.zeros(len(wind_range))

for i in range(0,len(wind_range)):
    e.set_v_w(wind_range[i])
    c.run_simulation(s, e)
    mean_powers[i] = c.cycle_power
    powers_out[i] = c.cycle_out_power
    powers_in[i] = c.cycle_in_power
    tether_forces_out[i] = c.tether_force_out
    tether_forces_in[i] = c.tether_force_in

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(wind_range, mean_powers*1e-3, label="cycle")
ax[0].plot(wind_range, powers_out, label="out")
ax[0].plot(wind_range, powers_in, label="in")
ax[0].legend()
ax[1].plot(wind_range, tether_forces_out, label="out")
ax[1].plot(wind_range, tether_forces_in, label="in")
ax[1].legend()

ax[0].set_ylabel('Power [kW]')
ax[1].set_ylabel('Force [kN]')
ax[1].set_xlabel('Wind speed [m/s]')

#%% Test run - Calculate the mass of the ground station
s = System()
e = Environment()
c = Cycle()
annual_calculation(s, e, graph=True)
gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max = get_operational_envelope(plot=False)
gs_mass = get_gs_mass(gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max)
system_mass = s.kite_mass + s.tether_mass + gs_mass + 5.3
print("System mass is {:.2f} kg.".format(arr_system_mass[0]))

        