import os
import sys
from mpi4py import MPI
from basix.ufl import element
from petsc4py import PETSc
from dolfinx import mesh, fem, io, log,plot,cpp,geometry
import ufl
import numpy as np
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
import cv2
from dolfinx import default_scalar_type
from dolfinx.io import VTKFile
import pyvista
import dolfinx
import gc
from scipy.interpolate import griddata
import netCDF4 as nc

import ipyparallel as ipp
import dolfinx.fem.petsc as dfx_petsc
import time
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import math

# for Guassian Random Field
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import multivariate_normal

from .settings import Problem_settings, Neches_problem_settings


class Cartesian_MosquitoSolver:

    def __init__(self,problem_settings):
        # set basic parameters
        self.diffusion_coefficient = problem_settings.diffusion_coefficient#18969 # m^2/day Need to edit it properly. 111111 for degree to meter conversion
        self.mu_1 = problem_settings.mu_1 #0.1177  # 1/day
        self.mu_2 = problem_settings.mu_2#0.0250  # 1/day
        self.oviation_rate = problem_settings.oviation_rate #34  # 1/day
        self.hatching_rate = problem_settings.hatching_rate # 0.24  # 1/day
        self.immobile_maturation_rate = problem_settings.immobile_maturation_rate#0.0625  # 1/day
        self.carrying_capacity = problem_settings.carrying_capacity#0.0590 # 1/m^2
        # self.constant_for_aquatic = problem_settings.constant_for_aquatic#0.0
        # self.constant_for_egg = problem_settings.constant_for_egg#0.0
        self.constant_for_mobile = problem_settings.constant_for_mobile#0.01

        self.earth_radius = problem_settings.earth_radius#6378206.4 # m
        self.period = problem_settings.period
        self.steps = problem_settings.steps
        self.nx = problem_settings.nx
        self.ny = problem_settings.ny
        self.save_path = problem_settings.save_path
        self.output_name = problem_settings.output_name
        self.delta_t = problem_settings.delta_t
        self.constant_for_mobile = self.constant_for_mobile*self.delta_t #todo: modify later
        self.Lon_min, self.Lon_max, self.Lat_min, self.Lat_max = problem_settings.Lon_min, problem_settings.Lon_max, problem_settings.Lat_min, problem_settings.Lat_max
        self.stag_path = problem_settings.stag_path 
        self.flag_advection = problem_settings.flag_advection
        self.flag_observation = problem_settings.flag_observation
        self.adcirc_param = problem_settings.adcirc_param

        self.flag_spinup = problem_settings.flag_spinup
        self.flag_separate_plots = problem_settings.flag_separate_plots
        self.spin_up_period= problem_settings.spin_up_period
        self.trap_constant = problem_settings.trap_constant
        


        
        # Print coordinates outside the domain
        if self.flag_observation:
            # Filter coordinates and identify those outside the domain
            self.stations = []
            outside_coordinates = []
            
            for lat, lon in problem_settings.stations:
                if self.Lat_min <= lat <= self.Lat_max and self.Lon_min <= lon <= self.Lon_max:
                    print(f"Station coordinates within the domain: ({lat}, {lon})")
                    self.stations.append((lat, lon))
                else:
                    outside_coordinates.append((lat, lon))

            if outside_coordinates:
                print("Coordinates outside the domain:")
                for coord in outside_coordinates:
                    print(coord)
            self.station_df = pd.DataFrame(columns=["station_number", "latitude", "longitude", "time", "mobile_population"])
            self.station_observe_days = problem_settings.station_observe_days

        if self.flag_advection:
            self.wind_param = problem_settings.wind_param
            self.wind_path = problem_settings.wind_path
            self.constant_for_wind = problem_settings.constant_for_wind


        #conversion
        self.center_lon = (self.Lon_min+self.Lon_max)/2
        self.center_lat = (self.Lat_min+self.Lat_max)/2
        self.x_min = self.earth_radius*np.cos(np.deg2rad(self.center_lat))*np.deg2rad(self.Lon_min-self.center_lon)
        self.x_max = self.earth_radius*np.cos(np.deg2rad(self.center_lat))*np.deg2rad(self.Lon_max-self.center_lon)
        self.y_min = self.earth_radius*np.deg2rad(self.Lat_min)
        self.y_max = self.earth_radius*np.deg2rad(self.Lat_max)
        # print("bounding box in meters:")
        # print([self.x_min,self.x_max,self.y_min,self.y_max])

        # produce the coordinates of mesh points
        x_coords = np.linspace(self.x_min, self.x_max, self.nx+1)
        y_coords = np.linspace(self.y_min, self.y_max, self.ny+1)
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        x_coordsdeg = np.rad2deg(x_coords/(np.cos(np.deg2rad(self.center_lat))*self.earth_radius)+np.deg2rad(self.center_lon))
        y_coordsdeg = np.rad2deg(y_coords/self.earth_radius)
        mesh_coords = np.array([x_coordsdeg, y_coordsdeg,x_coords,y_coords])
        

        # initialize domain
        self.domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([self.x_min, self.y_min]), np.array([self.x_max, self.y_max])],
                               [self.nx, self.ny], mesh.CellType.triangle)
        self.x = ufl.SpatialCoordinate(self.domain)

        # Define the function space
        self.P1 = element("Lagrange", self.domain.ufl_cell(), 1)
        self.V = fem.FunctionSpace(self.domain, ufl.MixedElement([self.P1, self.P1, self.P1]))

        # Define test and trial functions
        self.v_1, self.v_2, self.v_3 = ufl.TestFunctions(self.V)
        self.uh = fem.Function(self.V)
        self.u_1, self.u_2, self.u_3 = ufl.split(self.uh)
        # Define previous time step functions
        self.u_n = fem.Function(self.V)
        self.u_n1, self.u_n2, self.u_n3 = self.u_n.split()

        # Create the bounding box tree for efficient spatial queries
        self.bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)

        if not os.path.exists(self.save_path+"/video/"):
            os.makedirs(self.save_path+"/video/")
            print(f"Directory '{self.save_path+"/video/"}' created.")
        else:
            print(f"Directory '{self.save_path+"/video/"}' exists.")

        if self.flag_observation:
            print("Save mesh points.")
            np.savetxt(self.save_path+"mesh_point.csv", mesh_coords, delimiter=",")

    def spin_up(self):
        if not self.flag_spinup:
            raise ValueError("Spin-up is not enabled. Set flag_spinup to True to perform spin-up.")
        print("Spin up started.")
        # 1) reflective boundary condition is by defualt for FEniCSx
        # 2) settings record the min and max values of the initial conditions for randomly generated u_n1, u_n2, u_n3
        self.settings = {'u_1_min_val': 0.0, 'u_1_max_val': 0.02, 
                    'u_2_min_val': 0.0, 'u_2_max_val': 0.02, 
                    'u_3_min_val': 0.0, 'u_3_max_val': 0.02}
        # 3) set initial condition for spin up
        np.random.seed(42)
        self.u_n1.interpolate(lambda x: self.return_random(x,self.settings['u_1_min_val'], self.settings['u_1_max_val']))
        self.u_n2.interpolate(lambda x: self.return_random(x,self.settings['u_2_min_val'], self.settings['u_2_max_val']))
        self.u_n3.interpolate(lambda x: self.return_random(x,self.settings['u_3_min_val'], self.settings['u_3_max_val']))
        print("Initial conditions set for spin-up.")
        # Time-stepping loop
        this_t = 0

        xdmf_path = self.save_path + self.output_name + "_spin_up.xdmf"
        with io.XDMFFile(MPI.COMM_WORLD, xdmf_path, "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.u_n1, 0.0)
            xdmf.write_function(self.u_n2, 0.0)
            xdmf.write_function(self.u_n3, 0.0)

        while this_t+self.delta_t <= self.spin_up_period:
            start_time = time.time()
            gc.collect()
            
            this_t += self.delta_t
            print(f"Spin Up Time: {this_t:.2f}")

            a1 = ((self.u_1 - self.u_n1) / self.delta_t) * self.v_1 * ufl.dx + \
                    (self.mu_1 * self.u_1 * self.v_1 + \
                    self.diffusion_coefficient * ufl.dot(ufl.grad(self.u_1), ufl.grad(self.v_1)) - \
                        self.immobile_maturation_rate * self.u_3 * self.v_1 ) * ufl.dx
            a2 = ((self.u_2 - self.u_n2) / self.delta_t) * self.v_2 * ufl.dx + (-self.oviation_rate * self.u_1 * self.v_2 + self.hatching_rate * self.u_2 * self.v_2) * ufl.dx
            a3 = (((self.u_3 - self.u_n3)/self.delta_t) * self.v_3 - self.hatching_rate * (1-self.u_3/self.carrying_capacity)*self.u_2*self.v_3+(self.mu_2+self.immobile_maturation_rate)*self.u_3*self.v_3)* ufl.dx


            F = a1 + a2 + a3 

            problem = NonlinearProblem(F, self.uh)
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1e-6  # Adjust relative tolerance
            solver.report = True

            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            ksp.setFromOptions()

            log.set_log_level(log.LogLevel.ERROR)
            n, converged = solver.solve(self.uh)
            assert (converged)            
            # core PDE solver ends here
            # Update the solution for the next time step
            self.uh.x.scatter_forward()
            self.u_n.x.array[:] = self.uh.x.array
            # save the solution to XDMF
            with io.XDMFFile(MPI.COMM_WORLD, xdmf_path, "a") as xdmf:
                xdmf.write_function(self.u_n1, this_t)
                xdmf.write_function(self.u_n2, this_t)
                xdmf.write_function(self.u_n3, this_t)
            

            print(f"Time taken for this step: {time.time() - start_time:.2f} seconds")
            gc.collect()
        
        print("Spin up completed.")
        print("File saved at: ", xdmf_path)



    def clear_fenics_cache(self):
        cache_dir = os.path.expanduser("~/.cache/fenics")
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))



    def return_constant(self,x,const):
        return (x[0] * 0 + const)
    
    
    def update_mobile(self,this_t):
        # new_TODO: modify the water_nc_path
        water_nc_path = self.stag_path+'time_index_'+str(int((this_t//((self.adcirc_param[0]/self.adcirc_param[1])*self.adcirc_param[2]))*self.adcirc_param[2]))+'/neches_stag.nc'
        water_file = nc.Dataset(water_nc_path, 'r')
        lon_data = np.array(water_file.variables['lon'][:])
        lat_data = np.array(water_file.variables['lat'][:])
        # lat-lon to meter conversion
        lon_data = self.earth_radius*np.cos(np.deg2rad(self.center_lat))*np.deg2rad(lon_data-self.center_lon)
        lat_data = self.earth_radius*np.deg2rad(lat_data)

        print_string1 = "lon: ["+str(np.min(lon_data))+", "+str(np.max(lon_data))+"] , lat: ["+str(np.min(lat_data))+", "+str(np.max(lat_data))+"]"


        # Read the actual water data (2D grid of water presence/amount)
        water_data = np.array(water_file.variables['stag_water'][:])

        water_data = water_data[::-1,::-1]

        

        lon_lat_grid = np.array(np.meshgrid(lon_data, lat_data)).reshape(2, -1).T  # Shape (n_points, 2)
        water_values_flat = np.flip(water_data.flatten())

        # Get the mesh coordinates (longitude, latitude) from the FEniCSx mesh
        mesh_coords_2d = self.domain.geometry.x[:,:2]  # This gives the mesh coordinates (array of shape [n_points, 2])
        mesh_lon_min = np.min(mesh_coords_2d[:,0])
        mesh_lon_max = np.max(mesh_coords_2d[:,0])
        mesh_lat_min = np.min(mesh_coords_2d[:,1])
        mesh_lat_max = np.max(mesh_coords_2d[:,1])
        print_string2 = "mesh lon: ["+str(mesh_lon_min)+", "+str(mesh_lon_max)+"] , mesh lat: ["+str(mesh_lat_min)+", "+str(mesh_lat_max)+"]"


        #Interpolate the water data onto the mesh coordinates
        interpolated_water = griddata(lon_lat_grid, water_values_flat, mesh_coords_2d, method='linear')
        #self.plot_grid(interpolated_water.reshape(self.nx+1,self.ny+1))


        # Set NaN values to 0
        interpolated_water = np.nan_to_num(interpolated_water, nan=0.0)

        # Create a FEniCSx function to store the interpolated water data
        V = dolfinx.fem.FunctionSpace(self.domain, ("CG", 1))
        water_function = dolfinx.fem.Function(V)
        water_function.vector.setArray(interpolated_water)
        #self.plot_check(this_t,water_function,'water',print_string1,print_string2)


        this_mixed_water = fem.Function(self.V)
        this_mixed_water.sub(0).interpolate(water_function)
        self.u_n.vector.array[:] = self.u_n.vector.array[:]+this_mixed_water.vector.array[:] * self.constant_for_mobile


    def update_wind_data(self, time_step):
        """
        Reads wind data from NetCDF, extracts values at a given time step,
        interpolates them onto the FEniCSx mesh, and converts them into a FEniCSx Function.

        Parameters:
            mesh (dolfinx.mesh.Mesh): The FEniCSx mesh.
            time_step (int): The time step index.

        Returns:
            (dolfinx.fem.Function, dolfinx.fem.Function): Interpolated u and v wind fields.
        """
        #file_path = "/mnt/data/ce622436bf596857e19dbf01718f868a.nc"
        dataset = xr.open_dataset(self.wind_path)

        # Convert mosquito simulation time to NetCDF time index
        nc_time_index = math.floor((time_step/self.steps)*self.period/self.wind_param[0]*self.wind_param[1])

        # print(nc_time_index)

        # Extract wind components for the given time index
        u10 = dataset['u10'].isel(valid_time=nc_time_index).values* 86400 #convert unit from m/s to m/day
        v10 = dataset['v10'].isel(valid_time=nc_time_index).values* 86400 #convert unit from m/s to m/day

        # print("min/max lon in NetCDF:", np.min(dataset['longitude'].values), np.max(dataset['longitude'].values))
        # print("min/max lat in NetCDF:", np.min(dataset['latitude'].values), np.max(dataset['latitude'].values))

        # Get spatial coordinates from NetCDF
        lon_data = dataset['longitude'].values
        lat_data = dataset['latitude'].values
        
        lon_meters = self.earth_radius * np.cos(np.deg2rad(self.center_lat)) * np.deg2rad(lon_data - self.center_lon)
        lat_meters = self.earth_radius * np.deg2rad(lat_data)

        # print("min/max lon:", np.min(lon_meters), np.max(lon_meters))
        # print("min/max lat:", np.min(lat_meters), np.max(lat_meters))

        # Create meshgrid for interpolation
        lon_lat_grid = np.array(np.meshgrid(lon_meters, lat_meters)).reshape(2, -1).T

        # Flatten wind data for interpolation
        u_values_flat = u10.flatten()
        v_values_flat = v10.flatten()

        # Get FEniCSx mesh coordinates
        mesh_coords_2d = self.domain.geometry.x[:, :2]

        # Interpolate wind data onto the FEniCSx mesh
        interpolated_u = griddata(lon_lat_grid, u_values_flat, mesh_coords_2d, method='linear')
        interpolated_v = griddata(lon_lat_grid, v_values_flat, mesh_coords_2d, method='linear')

        # Replace NaNs with zero
        interpolated_u = np.nan_to_num(interpolated_u, nan=0.0)
        interpolated_v = np.nan_to_num(interpolated_v, nan=0.0)

        # Define vector function space
        V = dolfinx.fem.VectorFunctionSpace(self.domain, ("CG", 1), dim=2)
        wind_function = fem.Function(V)

        # Assign interpolated values to wind function
        #wind_function.x.array[:len(interpolated_u)] = np.column_stack((interpolated_u, interpolated_v)).flatten()
        num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        wind_values = np.zeros((num_dofs,))
        wind_values[0::2] = interpolated_u  # Assign U components
        wind_values[1::2] = interpolated_v  # Assign V components

        wind_function.x.array[:] = wind_values

        return wind_function


    def get_mesh_coords_from_latlon(self, latitude, longitude):
        """Convert geographic coordinates to mesh coordinates."""
        x = self.earth_radius * np.cos(np.deg2rad(self.center_lat)) * np.deg2rad(longitude - self.center_lon)
        y = self.earth_radius * np.deg2rad(latitude)
        return x, y

    def record_stations_at_time(self, stations, time):
        """Record mobile population at each station for a given time and update station_df."""
        points = np.array([self.get_mesh_coords_from_latlon(lat, lon) + (0.0,) for lat, lon in stations])

            
        # Find cells containing each station point
        cell_candidates = geometry.compute_collisions_points(self.bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points)
        
        for station_num, (point, (lat, lon)) in enumerate(zip(points, stations)):
            # Check if there is a cell containing this point
            cell_indices = colliding_cells.links(station_num)
            if len(cell_indices) > 0:
                cell_index = cell_indices[0]
                
                # Evaluate the mobile population at this point in the correct cell
                mobile_population = self.u_n.sub(0).eval([point], [cell_index])
                
                # Create a DataFrame for the new row and use pd.concat
                new_row = pd.DataFrame([{
                    "station_number": station_num,
                    "latitude": lat,
                    "longitude": lon,
                    "time": time,
                    "mobile_population": mobile_population[0]
                }])

                # Concatenate the new row to the existing DataFrame
                self.station_df = pd.concat([self.station_df, new_row], ignore_index=True)
            else:
                print(f"Warning: Point {point} (station {station_num}) is outside any cell.")

    def return_random(self, x, min_val, max_val, length_scale=500, flag_white_noise = False):
        """
        Generate a Gaussian Random Field across the spatial domain.
        To speed it up, we first generate a coarse grid and then interpolate to the fine mesh.
        
        Parameters:
            x (np.ndarray): Spatial points of shape (dim, n_points)
            min_val, max_val (float): Used for normalization
            length_scale (float): Correlation length of the GRF in mesh units

        Returns:
            np.ndarray: Sampled GRF values at mesh nodes
        """
        print("start to generate random field...")
        np.random.seed(np.random.randint(1000))
        if flag_white_noise:
            return np.random.uniform(min_val, max_val, len(x[0]))
        # 1. Build coarse grid in x and y
        print("start to build coarse grid...")
        x_min, x_max = x[0].min(), x[0].max()
        y_min, y_max = x[1].min(), x[1].max()

        coarse_nx0 = int(np.ceil((self.x_max - self.x_min) / length_scale))
        coarse_ny0 = int(np.ceil((self.y_max - self.y_min) / length_scale))

        coarse_nx = max(5, min(100, coarse_nx0))
        coarse_ny = max(5, min(100, coarse_ny0))

        print('advised coarse grid size:',coarse_nx0,coarse_ny0)
        print('applied grid size:',coarse_nx,coarse_ny)

        x_coarse = np.linspace(x_min, x_max, coarse_nx)
        y_coarse = np.linspace(y_min, y_max, coarse_ny)
        Xc, Yc = np.meshgrid(x_coarse, y_coarse)
        coarse_coords = np.column_stack([Xc.ravel(), Yc.ravel()])

        # 2. Build covariance matrix for coarse grid
        print("start to build covariance matrix...")
        kernel = C(1.0) * RBF(length_scale)
        cov_matrix = kernel(coarse_coords)
        coarse_sample = multivariate_normal.rvs(mean=np.zeros(coarse_coords.shape[0]), cov=cov_matrix)

        # 3. Interpolate to fine mesh
        print("start to interpolate to fine mesh...")
        fine_coords = np.stack([x[0], x[1]], axis=1)
        fine_sample = griddata(coarse_coords, coarse_sample, fine_coords, method='cubic')

        # 4. Normalize to [min_val, max_val]
        fine_sample = np.nan_to_num(fine_sample, nan=np.mean(coarse_sample))
        fine_sample = (fine_sample - fine_sample.min()) / (fine_sample.max() - fine_sample.min())
        return fine_sample * (max_val - min_val) + min_val
    
    def save_station_data(self):
        """Save station data to a CSV file at the end of the run."""
        csv_file1 = os.path.join(self.save_path, f"{self.output_name}_stations_data.csv")
        self.station_df.to_csv(csv_file1, index=False)
        print(f"Data saved to {csv_file1}")
        csv_file2 = os.path.join(self.save_path, f"{self.output_name}_stations_time_integration.csv")
        integrated_results = {}
        for station_num in self.station_df["station_number"].unique():
            station_data = self.station_df[self.station_df["station_number"] == station_num]
            station_data = station_data.sort_values("time")  # Ensure sorted by time for integration
            
            # Time and mobile population values
            time_values = station_data["time"].values
            mobile_population_values = station_data["mobile_population"].values
            
            # Perform trapezoidal integration over time
            integrated_value = np.trapz(mobile_population_values, time_values)
            integrated_results[station_num] = integrated_value*self.trap_constant

        # Save all integrated results in a single CSV
        integrated_df = pd.DataFrame(list(integrated_results.items()), columns=["station_number", "integrated_mobile_population"])
        integrated_df.to_csv(csv_file2, index=False)
        print(f"Integrated data saved to {csv_file2}")

    def plot_separate(self,this_t,item,item_name):
        """
        Plots the specified item (u_n1, u_n2, or u_n3) at a given time step and saves it as a heatmap PNG file.
        Parameters:
            item (int): The index of the item to plot (0, 1, or 2).
            item_name (str): The name of the item to plot ("u_n1", "u_n2", or "u_n3").
            time_step (int or float): The current time step.
        """
        font_size = 10
        title_text0 = "Time: "+str(this_t)
        title_text2 = "nx "+str(self.nx)+", ny "+str(self.ny)+", Lon: ["+str(self.Lon_min)+", "+str(self.Lon_max)+"] , Lat: ["+str(self.Lat_min)+", "+str(self.Lat_max)+"]"
        title_text3 = "period "+str(self.period)+", steps "+str(self.steps)
        num_cells = self.domain.topology.index_map(self.domain.topology.dim).size_local
        cell_entities = np.arange(num_cells, dtype=np.int32)  # Corrected dtype to int32

        # Create VTK mesh
        args = dolfinx.plot.vtk_mesh(self.domain, self.domain.topology.dim, cell_entities)
        #print(args)
        grid = pyvista.UnstructuredGrid(*args)

        this_data = self.u_n.sub(item).collapse().x.array.real
        plotter = pyvista.Plotter(title="Mixed")
        grid.point_data[item_name] = this_data
        grid.set_active_scalars(item_name)
        plotter.add_mesh(grid, show_scalar_bar=True, show_edges=False,scalar_bar_args={'title_font_size': font_size, 'label_font_size': font_size,'width':0.8,"position_x":0.1})
        plotter.add_text(item_name, position='upper_left',font_size=font_size)       
        plotter.add_text(title_text0, position=(0.01, 0.93), font_size=font_size - 1, viewport=True, color="black")
        plotter.add_text(title_text2, position=(0.01, 0.90), font_size=font_size - 1, viewport=True, color="black")
        plotter.add_text(title_text3, position=(0.01, 0.87), font_size=font_size - 1, viewport=True, color="black")
        plotter.view_xy()
        plotter.render()
        plot_file = f"{self.save_path}/video/{self.output_name}_frame_{int(this_t / self.delta_t):05d}_{item_name}.png"
        plotter.screenshot(plot_file,window_size=[800,800])
        plotter.close()

    def plot_wind_field(self, wind_function, this_t):
        """
        Plots the wind velocity field (u, v) at a given time step and saves it as a heatmap PNG file.

        Parameters:
            wind_function (dolfinx.fem.Function): The FEniCSx wind function (VectorFunctionSpace).
            time_step (int or float): The current time step.
        """
        # Ensure save directory exists
        wind_save_path = os.path.join(self.save_path, "wind_plots")
        os.makedirs(wind_save_path, exist_ok=True)

        # Extract mesh coordinates
        coords = wind_function.function_space.mesh.geometry.x[:, :2]  # Get (x, y) coordinates

        # Ensure wind_function data is correctly reshaped
        try:
            wind_values = wind_function.x.array.reshape(-1, 2)  # Reshape to (num_points, 2)
            u_values = wind_values[:, 0]  # U component
            v_values = wind_values[:, 1]  # V component
        except ValueError as e:
            print(f"Error reshaping wind function data: {e}")
            return

        # Ensure we have valid data
        if coords.shape[0] != u_values.shape[0]:
            print("Mismatch between mesh coordinates and wind data points!")
            return

        # Define structured grid for interpolation
        x_unique = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
        y_unique = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
        X, Y = np.meshgrid(x_unique, y_unique)

        # Interpolate wind data onto a regular grid
        U_grid = griddata(coords, u_values, (X, Y), method='linear', fill_value=0)
        V_grid = griddata(coords, v_values, (X, Y), method='linear', fill_value=0)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Define the correct extent for the plots
        extent = [coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()]

        # Plot U-component (heatmap)
        im1 = axes[0].imshow(U_grid, aspect='auto', origin='lower', cmap='coolwarm', extent=extent)
        axes[0].set_title("U-component (East-West Wind)")
        axes[0].set_xlabel("Longitude (m)")
        axes[0].set_ylabel("Latitude (m)")
        fig.colorbar(im1, ax=axes[0])

        # Plot V-component (heatmap)
        im2 = axes[1].imshow(V_grid, aspect='auto', origin='lower', cmap='coolwarm', extent=extent)
        axes[1].set_title("V-component (North-South Wind)")
        axes[1].set_xlabel("Longitude (m)")
        fig.colorbar(im2, ax=axes[1])

        # Save the figure
        filename = os.path.join(wind_save_path, f"wind_step_{this_t:04f}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved wind field heatmap: {filename}")

    def make_video(self,item_name,dir_name):
        """
        only if you produce separate plots with self.plot_separate, you can make a video with this function

        Parameters:
        dir_name: the path to the directory where the images are saved
        item_name: the identifier of the item you want to make a video for, items in the folder should have distinct identifiers
        """
        # Set the paths and parameters
        input_path = self.save_path + "/"+dir_name+"/"
        output_path = self.save_path+item_name+'_video.mp4'
        frame_rate = 24

        # Get the list of image files
        image_files = sorted([f for f in os.listdir(input_path) if f.endswith(item_name+'.png')])

        # Read the first image to get the frame size
        first_image = cv2.imread(os.path.join(input_path, image_files[0]))
        height, width, layers = first_image.shape
        frame_size = (width, height)

        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size)

        # Write each frame to the video
        for image_file in image_files:
            img = cv2.imread(os.path.join(input_path, image_file))
            video_writer.write(img)

        # Release the VideoWriter
        video_writer.release()

    def solve_mosquito(self):
        """
        Main function to solve the mosquito PDE problem.
        You can call this function after you initialize the problem (and finish spin up).
        """
        if not self.flag_spinup:
            self.u_n1.interpolate(lambda x: self.return_constant(x,0))
            self.u_n2.interpolate(lambda x: self.return_constant(x,0))
            self.u_n3.interpolate(lambda x: self.return_constant(x,0))

        # Prepare boundary condition - todo
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        uD = fem.Function(self.V)

        # Define the zero in the boundary condition
        zero_boundary = fem.Function(self.V)
        zero_boundary.sub(0).interpolate(lambda x: self.return_constant(x,0))
        zero_boundary.sub(1).interpolate(lambda x: self.return_constant(x,0))
        zero_boundary.sub(2).interpolate(lambda x: self.return_constant(x,0))
        bc = fem.dirichletbc(zero_boundary, boundary_dofs)

        # Redirect output to a file
        rank = MPI.COMM_WORLD.rank
        # self.redirect_output(rank)

        # Time-stepping loop
        this_t = 0



        xdmf_path = self.save_path + self.output_name + ".xdmf"
        with io.XDMFFile(MPI.COMM_WORLD, xdmf_path, "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.u_n1, 0.0)
            xdmf.write_function(self.u_n2, 0.0)
            xdmf.write_function(self.u_n3, 0.0)
        
        # Plot images for making videos later
        if self.flag_separate_plots:
            self.plot_separate(this_t,0,'mobile')
            self.plot_separate(this_t,1,'egg')
            self.plot_separate(this_t,2,'aquatic')
        
        self.update_mobile(this_t)

        if self.flag_advection:
            wind_function = self.update_wind_data(this_t)



        while this_t+self.delta_t <= self.period:
            start_time = time.time()
            gc.collect()
            
            this_t += self.delta_t
            print(f"Time: {this_t:.2f}")

            # Core PDE solver starts here
            if self.flag_advection:
                if self.flag_separate_plots:
                    self.plot_wind_field(wind_function, this_t)
                a1 = ((self.u_1 - self.u_n1) / self.delta_t) * self.v_1 * ufl.dx + \
                (self.mu_1 * self.u_1 * self.v_1 + \
                 self.diffusion_coefficient * ufl.dot(ufl.grad(self.u_1), ufl.grad(self.v_1)) - \
                    self.immobile_maturation_rate * self.u_3 * self.v_1 + \
                        self.constant_for_wind * ufl.div(wind_function * self.u_1) * self.v_1) * ufl.dx
            else:
                a1 = ((self.u_1 - self.u_n1) / self.delta_t) * self.v_1 * ufl.dx + \
                    (self.mu_1 * self.u_1 * self.v_1 + \
                    self.diffusion_coefficient * ufl.dot(ufl.grad(self.u_1), ufl.grad(self.v_1)) - \
                        self.immobile_maturation_rate * self.u_3 * self.v_1 ) * ufl.dx
            a2 = ((self.u_2 - self.u_n2) / self.delta_t) * self.v_2 * ufl.dx + (-self.oviation_rate * self.u_1 * self.v_2 + self.hatching_rate * self.u_2 * self.v_2) * ufl.dx
            a3 = (((self.u_3 - self.u_n3)/self.delta_t) * self.v_3 - self.hatching_rate * (1-self.u_3/self.carrying_capacity)*self.u_2*self.v_3+(self.mu_2+self.immobile_maturation_rate)*self.u_3*self.v_3)* ufl.dx


            F = a1 + a2 + a3 

            problem = NonlinearProblem(F, self.uh, bcs=[bc])
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1e-6  # Adjust relative tolerance
            solver.report = True

            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            ksp.setFromOptions()

            log.set_log_level(log.LogLevel.ERROR)
            n, converged = solver.solve(self.uh)
            assert (converged)
            
            # core PDE solver ends here

            # Update the solution for the next time step
            self.uh.x.scatter_forward()
            self.u_n.x.array[:] = self.uh.x.array

            # Plot images for making videos later
            if self.flag_separate_plots:
                self.plot_separate(this_t,0,'mobile')
                self.plot_separate(this_t,1,'egg')
                self.plot_separate(this_t,2,'aquatic')

            if self.flag_observation:
                if (this_t<self.station_observe_days[1] and this_t>self.station_observe_days[0]):
                    print("record station information at time: ",this_t)
                    self.record_stations_at_time(self.stations, this_t)

            # save the solution to XDMF
            with io.XDMFFile(MPI.COMM_WORLD, xdmf_path, "a") as xdmf:
                xdmf.write_function(self.u_n1, this_t)
                xdmf.write_function(self.u_n2, this_t)
                xdmf.write_function(self.u_n3, this_t)
                
            # Update the mobile population and wind data
            self.update_mobile(this_t)
            if self.flag_advection:
                wind_function = self.update_wind_data(this_t)
            print(f"Time taken for this step: {time.time() - start_time:.2f} seconds")


        # Save the station data to a CSV file
        if self.flag_observation:
            self.save_station_data()
        gc.collect()

        # Make videos from the plotted images

        if self.domain.comm.rank == 0:
            if self.flag_separate_plots:
                self.make_video('mobile',"video")
                self.make_video('egg',"video")
                self.make_video('aquatic',"video")
                if self.flag_advection:
                    self.make_video('',"wind_plots")


   