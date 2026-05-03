# GR-NBody: Post-Newtonian N-Body Integrator

GR-NBody is a high-performance, modular N-body integration codebase written in Python. It is designed to simulate the orbital dynamics of multi-body systems under both pure Newtonian gravity and first-order Post-Newtonian (1PN) general relativistic approximations.

This project was developed to quantify the deviation between Newtonian and 1PN orbits, such as the anomalous perihelion precession of Mercury, and to test the long-term stability of compact multi-body systems.

## Features
* **Dual Physics Engines:** Toggle between standard Newtonian gravity and 1PN general relativistic corrections.
* **Multiple Integrators:** Choose between 4th-Order Runge-Kutta (RK4) for high-accuracy short-term integrations, or Symplectic Leapfrog for long-timescale energy conservation.
* **High Performance:** Core $\mathcal{O}(N^2)$ force calculations are JIT-compiled using `numba` for C-like execution speeds.
* **Data-Driven Configuration:** Simulations are entirely controlled via lightweight `.json` input files—no need to modify source code to change initial conditions.
* **Automated Diagnostics:** Automatically generates trajectory plots and fractional energy error diagnostics to ensure numerical stability.

---

## Directory Structure

```text
GR-NBody/
├── configs/                   # JSON configuration files for initial conditions
│   └── config.json            
├── outputs/                   # Directory for generated data and plots
├── main.py                    # Command-line interface and simulation orchestrator
├── physics.py                 # Numba-optimized force calculations
├── integrators.py             # RK4 and Leapfrog numerical integration steps
├── analysis.py                # Energy calculations and plotting tools
├── tests.py                   # Automated physics test suite
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

---

## Installation

**Prerequisites:** Python 3.8+ is required. 

1. Clone the repository:
   ```bash
   git clone [https://github.com/ASTRX470-S26/final-ASTR5470-Anthony-Wang.git](https://github.com/ASTRX470-S26/final-ASTR5470-Anthony-Wang.git)
   cd GR-NBody
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Main dependencies: `numpy`, `numba`, `matplotlib`)*

---

## Usage

To run a simulation, execute `main.py` and pass the path to your desired JSON configuration file:

```bash
python main.py configs/config.json
```

**Outputs:**
Upon completion, the simulation will save the following files to your working directory (or the `outputs/` folder if configured):
* `trajectory_data.csv`: Time-series state vectors for all bodies.
* `trajectories.png`: A 2D plot of the orbital paths.
* `energy_error.png`: A diagnostic plot of the fractional energy drift.

---

## Running the Tests

This codebase includes an automated test suite to verify physical accuracy (Energy Conservation, Kepler's Third Law, and 1PN Relativistic Precession). 

To run the tests:
```bash
python tests.py
```

---

## Documentation

For a comprehensive breakdown of the physics background, integration math, and a detailed guide on how to format your `.json` configuration files, please visit the **[Project Wiki]([https://github.com/ASTRX470-S26/final-ASTR5470-Anthony-Wang/wiki])**.
