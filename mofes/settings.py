import os
class Problem_settings:
    def __init__(self, diffusion_coefficient, mu_1, mu_2, oviation_rate, 
                         hatching_rate, immobile_maturation_rate, carrying_capacity, 
                         constant_for_aquatic, constant_for_egg, constant_for_mobile, 
                         earth_radius, period, steps, nx, ny, save_path, 
                         output_name, bounding_box, stag_path, adcirc_param, flag_spinup, wind_param,
                         flag_advection,flag_observation,
                        wind_path,constant_for_wind,stations, station_observe_days,flag_separate_plots,spin_up_period=0.0):
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.oviation_rate = oviation_rate
        self.hatching_rate = hatching_rate
        self.immobile_maturation_rate = immobile_maturation_rate
        self.carrying_capacity = carrying_capacity
        self.constant_for_aquatic = constant_for_aquatic
        self.constant_for_egg = constant_for_egg
        self.constant_for_mobile = constant_for_mobile
        self.earth_radius = earth_radius
        self.period = period
        self.steps = steps
        self.nx = nx
        self.ny = ny
        self.save_path = save_path
        self.output_name = output_name
        self.delta_t = period / steps
        self.Lon_min, self.Lon_max, self.Lat_min, self.Lat_max = bounding_box
        self.stag_path = stag_path 
        self.flag_advection = flag_advection
        self.flag_observation = flag_observation
        self.flag_spinup = flag_spinup
  
        self.adcirc_param = adcirc_param
        self.flag_separate_plots = flag_separate_plots
        self.spin_up_period = spin_up_period
        self.trap_constant = 400.0

        
        
        

        # Filter coordinates and identify those outside the domain
        if self.flag_observation:
            if not stations:
                raise ValueError("Observation is enabled but no stations provided.")
            self.station_observe_days = station_observe_days
            self.stations = stations
            self.stations = []
            outside_coordinates = []
            
            for lat, lon in stations:
                if self.Lat_min <= lat <= self.Lat_max and self.Lon_min <= lon <= self.Lon_max:
                    self.stations.append((lat, lon))
                else:
                    outside_coordinates.append((lat, lon))
            
            # Print coordinates outside the domain
            if outside_coordinates:
                print("Coordinates outside the domain:")
                for coord in outside_coordinates:
                    print(coord)
        
        # Raise exception if we are including advection but no wind path provided
        if self.flag_advection:
            if not wind_path:
                raise ValueError("Advection is enabled but no wind path provided.")
            if not os.path.exists(wind_path):
                raise FileNotFoundError(f"Wind path '{self.wind_path}' does not exist.")
            self.wind_path = wind_path
            self.wind_param = wind_param
            self.constant_for_wind = constant_for_wind


class Neches_problem_settings(Problem_settings):
    # design two switchs for advection and station observation
    def __init__(self, diffusion_coefficient, mu_1, mu_2, oviation_rate, hatching_rate, 
                 immobile_maturation_rate, carrying_capacity, constant_for_mobile, period, steps, 
                 nx, ny, save_path, output_name, stag_path, flag_spinup,
                 flag_advection=False,flag_observation=False,
                 wind_path=None,constant_for_wind=None,stations=None, station_observe_days=None,flag_separate_plots=False):
        earth_radius = 6378206.4 # m
        bounding_box = [-94.12, -93.80, 29.90, 30.20]
        adcirc_param = [4, 384, 20] #adcirc period(day), adcirc output timestep, stagnated width
        wind_param = [4, 120] #wind period(day), wind output width(number of time steps)
        constant_for_aquatic = 0.0
        constant_for_egg = 0.0
        spin_up_period = 4.0 #days, roughly 3-6 times the typical lifecycle of mosquitoes (7-14 days)
        super().__init__(diffusion_coefficient, mu_1, mu_2, oviation_rate, 
                         hatching_rate, immobile_maturation_rate, carrying_capacity, 
                         constant_for_aquatic, constant_for_egg, constant_for_mobile, 
                         earth_radius, period, steps, nx, ny, save_path, 
                         output_name, bounding_box, stag_path, adcirc_param,flag_spinup,wind_param,
                         flag_advection,flag_observation,
                        wind_path,constant_for_wind,stations, station_observe_days,flag_separate_plots,spin_up_period)

