# [MOFES](https://github.com/largeseabass/MOFES.git)
Mosquito Finite Element Solver


MOFES is a high-performance PDE-based simulation package for modeling mosquito population dynamics post-tropical-cyclones under complex environmental influences such as wind and water stagnation. It is built using [FEniCSx](https://fenicsproject.org/) and is designed to support public health applications and environmental forecasting.

## 🚀 Features

- ✅ Models mosquito life stages (egg, aquatic, mobile) using coupled PDEs
- 🌬️ Supports advection with wind data
- 🌊 Integrates water stagnation maps (e.g., post-hurricane flooding)
- 📈 Station-level observation and time integration output
- 🧪 Gaussian random field initial conditions for spin-up
- ⚙️ Built on FEniCSx, PETSc, and PyVista for performance and visualization

## Before you start

### 1. Clone the Repository

In terminal, under your favorite directory:
```
git clone https://github.com/largeseabass/MOFES.git
cd MOFES
```

### 2. Set up Conda Environment

In terminal, navigate to MOFES folder, then:
```
conda create -n mofes-env python=3.12
conda activate mofes-env
conda install -c conda-forge fenics-dolfinx mpich petsc4py mpi4py pyvista matplotlib opencv netcdf4 scipy pandas xarray ipyparallel scikit-learn 

```

### 3. Install package

```
pip install -e . 
```

And you can test if everything is installed successfully by running the following line in the terminal:
```
python -c "import mofes; print('MOFES loaded successfully')"
```

## Prepare Input Files

**Stagnated water netCDF files** Go to /kalpana-dependent directory and follow the instruction in its README.md

**Surface Wind Field**  Go to [ERA5 Reanalysis Product](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) to download 10 m u-component of wind and 10 m v-component of wind.

## MOFES Example

You can find one example of a 4-day simulation for the Neches River Test Case after Hurricane Harvey 2017 landfall at /examples/mosquito_problem_spinup_wind.ipynb. All the required data is shared under the /data folder.

## 📄 License

MIT License. See [`LICENSE`](LICENSE) for details.


## Author

Liting Huang
Graduate Student at UT Austin when this code was created
GitHub: @largeseabass
Email: litinghuang42@gmail.com

[1] H. Hersbach, B. Bell, P. Berrisford, G. Biavati, A. Hor´anyi, J. Mu˜noz Sabater, J. Nicolas, C. Peubey, R. Radu, I. Rozum, D. Schepers, A. Simmons, C. Soci, D. Dee, and J-N. Th´epaut. Era5 hourly data on single levels from 1940 to present, 2023. Accessed on 18-02-2025