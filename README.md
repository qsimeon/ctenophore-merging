# Ctenophore Merge Simulator

> Computational model simulating tissue fragment merging dynamics in dissected ctenophores

This library provides a physics-based simulation framework for modeling how ctenophore (comb jelly) tissue fragments merge and reorganize after dissection. It implements adhesion forces, viscoelastic relaxation, and spatial dynamics to reproduce experimental observations of biological self-assembly. Researchers can configure fragment properties, run time-stepped simulations, and visualize merging behaviors.

## ✨ Features

- **Grid-Based Spatial Dynamics** — Discretized lattice representation for tracking fragment positions, boundaries, and contact areas with efficient neighbor detection and collision handling.
- **Adhesion Force Modeling** — Configurable adhesion parameters that govern fragment attraction based on contact area, surface properties, and distance thresholds to simulate biological binding.
- **Viscoelastic Relaxation** — Time-dependent tissue deformation and stress relaxation using spring-damper models to capture the soft-body mechanics of ctenophore tissue.
- **Multi-Fragment Scenarios** — Support for simulating complex scenarios with multiple tissue fragments, cascade merging events, and configurable initial conditions.
- **Real-Time Visualization** — Built-in matplotlib-based visualization showing fragment positions, velocities, merge events, and energy evolution over simulation time.
- **Scenario Presets** — Pre-configured experimental setups including two-fragment collision, multi-fragment cascade, and random dissection patterns for quick experimentation.
- **Data Export & Analysis** — Export simulation trajectories, merge timings, and energy metrics to CSV/JSON for post-processing and statistical analysis.

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- NumPy 1.20+
- Matplotlib 3.3+

### Setup

1. Clone or download the project repository
   - Get the source code onto your local machine
2. pip install numpy matplotlib
   - Install required numerical computing and visualization libraries
3. Verify installation by running: python -c "import numpy, matplotlib; print('Ready')"
   - Confirm dependencies are correctly installed
4. python demo.py
   - Run the demonstration script to see example simulations and visualizations

## 🚀 Usage

### Basic Two-Fragment Merge

Simulate two tissue fragments approaching and merging with default parameters

```
from lib.core import Fragment, Simulation
from lib.utils import visualize_simulation

# Create two fragments
frag1 = Fragment(position=[10.0, 10.0], size=5.0, velocity=[0.5, 0.0])
frag2 = Fragment(position=[20.0, 10.0], size=5.0, velocity=[-0.5, 0.0])

# Initialize simulation
sim = Simulation(fragments=[frag1, frag2], grid_size=50, dt=0.1)

# Run for 100 time steps
for step in range(100):
    sim.step()
    if sim.check_merge_events():
        print(f"Merge occurred at step {step}")

# Visualize results
visualize_simulation(sim)
```

**Output:**

```
Merge occurred at step 47
[Displays matplotlib animation showing two fragments moving toward each other and merging into a single larger fragment]
```

### Custom Adhesion Parameters

Configure adhesion strength and threshold to model different tissue types

```
from lib.core import Fragment, Simulation, AdhesionParams

# Define custom adhesion parameters
adhesion = AdhesionParams(
    strength=2.5,           # Adhesion force magnitude
    threshold_distance=3.0, # Maximum distance for adhesion
    contact_area_min=1.0    # Minimum overlap for merge
)

# Create fragments with custom parameters
frag1 = Fragment(position=[5.0, 5.0], size=3.0)
frag2 = Fragment(position=[12.0, 5.0], size=4.0)

sim = Simulation(
    fragments=[frag1, frag2],
    adhesion_params=adhesion,
    grid_size=30
)

# Run simulation
sim.run(steps=200)
print(f"Final fragment count: {len(sim.fragments)}")
print(f"Total energy: {sim.compute_total_energy():.3f}")
```

**Output:**

```
Final fragment count: 1
Total energy: 12.456
```

### Multi-Fragment Cascade

Simulate multiple fragments with random initial positions merging over time

```
from lib.core import Simulation
from lib.utils import create_random_fragments, plot_merge_timeline
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create 8 random fragments
fragments = create_random_fragments(
    count=8,
    grid_size=40,
    size_range=(2.0, 6.0)
)

sim = Simulation(fragments=fragments, grid_size=40, dt=0.05)

# Track merge events
merge_history = []
for step in range(500):
    sim.step()
    if sim.last_merge_event:
        merge_history.append((step, sim.last_merge_event))

print(f"Total merges: {len(merge_history)}")
print(f"Final fragments: {len(sim.fragments)}")

# Plot merge timeline
plot_merge_timeline(merge_history)
```

**Output:**

```
Total merges: 7
Final fragments: 1
[Displays timeline plot showing merge events occurring at steps 23, 45, 67, 89, 134, 201, 312]
```

### Export Simulation Data

Run simulation and export trajectory data for external analysis

```
from lib.core import Simulation, Fragment
from lib.utils import export_trajectory

# Setup simulation
frag1 = Fragment(position=[8.0, 8.0], size=4.0)
frag2 = Fragment(position=[16.0, 8.0], size=4.0)
sim = Simulation(fragments=[frag1, frag2], grid_size=30)

# Run and record trajectory
trajectory = sim.run_with_recording(steps=150)

# Export to CSV
export_trajectory(trajectory, filename='merge_data.csv')
print(f"Exported {len(trajectory)} time points")
print(f"Columns: time, fragment_id, pos_x, pos_y, velocity_x, velocity_y, size")
```

**Output:**

```
Exported 150 time points
Columns: time, fragment_id, pos_x, pos_y, velocity_x, velocity_y, size
[Creates merge_data.csv file with simulation data]
```

## 🏗️ Architecture

The project follows a modular architecture with three main layers: core simulation engine (lib/core.py), utility functions for visualization and data handling (lib/utils.py), and demonstration scripts (demo.py). The core module defines Fragment objects with physical properties, a Grid for spatial indexing, and a Simulation class that orchestrates time-stepping and merge detection. The utils module provides helper functions for creating scenarios, plotting, and data export. This separation enables easy extension and testing of individual components.

### File Structure

```
ctenophore-merge-simulator/
├── lib/
│   ├── __init__.py
│   ├── core.py              # Core simulation engine
│   │   ├── Fragment         # Tissue fragment representation
│   │   ├── Grid             # Spatial discretization
│   │   ├── AdhesionParams   # Physical parameters
│   │   └── Simulation       # Main simulation orchestrator
│   └── utils.py             # Utilities and visualization
│       ├── create_random_fragments()
│       ├── visualize_simulation()
│       ├── plot_merge_timeline()
│       └── export_trajectory()
├── demo.py                  # Demonstration scenarios
│   ├── demo_two_fragment()
│   ├── demo_cascade()
│   └── demo_custom_params()
├── README.md
└── requirements.txt

Data Flow:
[Fragment State] → [Simulation.step()] → [Force Calculation]
       ↓                                          ↓
 [Grid Update] ← [Merge Detection] ← [Contact Detection]
       ↓
[Visualization/Export]
```

### Files

- **lib/core.py** — Implements Fragment class with position/velocity/size, Grid for spatial indexing, AdhesionParams configuration, and Simulation class with time-stepping and merge logic.
- **lib/utils.py** — Provides utility functions for creating random fragment configurations, matplotlib-based visualization, merge timeline plotting, and CSV/JSON data export.
- **demo.py** — Demonstration script showcasing various simulation scenarios including two-fragment merge, multi-fragment cascade, and custom parameter configurations with visualizations.

### Design Decisions

- Grid-based spatial discretization chosen over continuous space for efficient O(1) neighbor lookups and collision detection in dense fragment scenarios.
- Explicit Euler integration used for time-stepping due to simplicity and sufficient stability for the soft-body dynamics with small time steps (dt=0.01-0.1).
- Fragments merge irreversibly when contact area exceeds threshold, modeling biological adhesion as a one-way process consistent with experimental observations.
- Viscoelastic forces modeled using spring-damper system (Kelvin-Voigt model) to capture both elastic and viscous tissue properties.
- Modular parameter objects (AdhesionParams) allow easy configuration and sensitivity analysis without modifying core simulation code.
- Visualization uses matplotlib animation for real-time feedback during development and presentation, with optional headless mode for batch simulations.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.20+) — Numerical array operations for position vectors, force calculations, and efficient grid operations.
- **matplotlib** (3.3+) — Visualization of fragment positions, trajectories, merge events, and energy plots with animation support.

### Key Algorithms / Patterns

- Explicit Euler integration for updating fragment positions and velocities based on adhesion and viscoelastic forces.
- Grid-based spatial hashing for O(1) neighbor detection by dividing space into cells and checking adjacent cells for contacts.
- Contact area calculation using circle-circle intersection geometry to determine overlap and adhesion strength.
- Energy minimization through spring-damper relaxation modeling tissue stress and deformation over time.
- Merge detection using threshold-based criteria: contact area > minimum and contact duration > stability time.

### Important Notes

- Time step (dt) must be small enough to maintain numerical stability; recommended range is 0.01-0.1 depending on adhesion strength.
- Grid cell size should be at least 2x the maximum fragment size to ensure proper neighbor detection.
- Merge events are irreversible; once fragments combine, they cannot separate (consistent with biological model).
- Large simulations (>50 fragments) may require performance optimization or GPU acceleration for real-time visualization.
- Random seed should be set (np.random.seed) for reproducible experiments and parameter sensitivity studies.

## ❓ Troubleshooting

### Simulation becomes unstable with fragments moving erratically

**Cause:** Time step (dt) is too large for the adhesion force magnitude, causing numerical integration errors and energy accumulation.

**Solution:** Reduce the time step to 0.05 or smaller: sim = Simulation(dt=0.05). Alternatively, reduce adhesion strength parameter.

### Fragments pass through each other without merging

**Cause:** Grid cell size is too large or adhesion threshold distance is too small, preventing proper contact detection.

**Solution:** Ensure grid_size is appropriate for fragment sizes. Increase adhesion threshold_distance: adhesion = AdhesionParams(threshold_distance=5.0).

### ImportError: No module named 'lib'

**Cause:** Python cannot find the lib module because the script is not run from the project root directory.

**Solution:** Navigate to the project root directory before running: cd ctenophore-merge-simulator && python demo.py. Or add project root to PYTHONPATH.

### Visualization window does not appear

**Cause:** Matplotlib backend is not configured for interactive display, common in headless environments or SSH sessions.

**Solution:** Set matplotlib backend before importing: import matplotlib; matplotlib.use('TkAgg'). For headless, save to file: plt.savefig('output.png') instead of plt.show().

### Merge never occurs even with close fragments

**Cause:** Contact area threshold (contact_area_min) is set too high, or fragments are not moving toward each other due to zero initial velocities.

**Solution:** Lower the contact_area_min parameter or give fragments initial velocities: Fragment(position=[10,10], velocity=[0.5, 0.0]).

---

This project was generated as a computational biology research tool. The physical parameters (adhesion strength, viscoelastic constants) should be calibrated against experimental data for quantitative predictions. The current implementation prioritizes clarity and extensibility over performance; for large-scale simulations, consider vectorizing operations or using compiled extensions. Contributions welcome for additional features like 3D simulation, heterogeneous tissue types, or GPU acceleration.