# CryoNet — Cryogenic Feed-System Network Solver

CryoNet is a Python-based steady-state fluid network solver developed to analyze cryogenic propellant feed systems.

The model calculates **mass flow, pressure losses, node pressures, and operating margins** through interconnected pipes, valves, injectors, and other fluid-system components. It includes separate LOX and liquid methane feed networks and evaluates their combined performance at the combustion chamber.

## Project Overview

The propulsion model is based on the **NASA Project Morpheus HD4-A LOX/methane engine** and uses published engine requirements to establish realistic operating conditions.

The solver uses:

* Mass conservation at each network node
* Pressure relationships across each branch
* Darcy-Weisbach pipe losses
* Valve loss coefficients
* Injector orifice flow
* Temperature-dependent cryogenic fluid properties
* Nonlinear numerical solving using SciPy
* Operating-envelope and requirement checks

The network formulation follows the same basic conservation structure used in NASA's **Generalized Fluid System Simulation Program (GFSSP)**.

## Key Capabilities

* Separate LOX and liquid methane feed-system models
* Pressure and mass-flow calculation throughout the network
* Mixture-ratio calculation
* Injector pressure-drop evaluation
* Cryogenic saturation-margin checks
* Valve-opening and operating-condition variation
* Automatic design-envelope checks
* Interactive browser dashboard
* CSV and plot export for further analysis

## Reference Design

The baseline model uses published Project Morpheus parameters including:

* **Engine:** NASA Project Morpheus HD4-A
* **Propellants:** LOX / Liquid Methane
* **Thrust:** 4,200 lbf
* **Specific Impulse:** 215 s
* **Throttle Ratio:** 4:1
* **Reference Tank Pressure:** 350 psia
* **Reference Chamber Pressure:** 250 psia

Where detailed flight-system geometry or component data was unavailable, engineering assumptions are explicitly identified in the model.

## Running the Solver

Install the required libraries:

```bash
pip install numpy scipy matplotlib
```

Launch the interactive dashboard:

```bash
python CryoNet_Interactive.py
```

Export static plots and CSV results:

```bash
python CryoNet_Interactive.py --export
```

## Tools

**Python · NumPy · SciPy · Matplotlib · Fluid Mechanics · Nonlinear Numerical Methods**

## Model Scope

CryoNet is a **steady-state, one-dimensional, single-phase engineering model** intended for system-level analysis and learning. It is not an exact replica of the Morpheus propulsion system and does not model combustion dynamics, transient behavior, or validated flight hardware.
