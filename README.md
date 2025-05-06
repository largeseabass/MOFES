# MOFES
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
conda env create -f environment.yml
conda activate mofes-env
```

### 3. Install package

```
pip install -e . 
```

### Optional: 4. running tests
```
pytest tests/
```

## Examples

You can find one example of a 4-day simulation for the Neches River Test Case after Hurricane Harvey 2017 landfall at /examples/mosquito_problem_spinup_wind.ipynb. All the required data is shared under the /data folder.


## Author

Liting Huang
Graduate Student at UT Austin when this code was created
GitHub: @largeseabass
Email: litinghuang42@gmail.com