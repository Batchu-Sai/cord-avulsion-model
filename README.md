<<<<<<< HEAD
# Spinal Cord Avulsion Model

A computational model for analyzing traumatic spinal cord avulsion injuries under high-speed loading conditions.

## Overview

This repository contains a validated mechanical model of the thoracic spinal cord system that simulates the dynamic response to traumatic loading. The model captures the relationship between applied moments, cord strain, and failure onset, providing insights into injury mechanisms and tissue sensitivity.

## Key Features

- **Quasi-static analysis**: Cord strain vs. applied moment relationships
- **Dynamic simulation**: Time-domain response to step loading
- **Failure prediction**: Automatic detection of strain threshold crossing
- **Parameter sensitivity**: Analysis of tissue and system parameter effects
- **Publication-ready figures**: High-quality plots for manuscript inclusion

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cord-avulsion-model
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Model Usage

```python
from spinal_model import SpinalCordModel

# Create model instance
model = SpinalCordModel()

# Run simulation
t, phi, phi_dot, strain = model.simulate(M_0=6.5, t_final=0.1)

# Calculate failure moment
M_fail = model.eps_fail * model.L_0 / model.r_c * model.K_eff
```

### Generate All Figures

```bash
python analyze_model.py
```

This will generate:
- `strain_vs_moment.png`: Quasi-static strain vs. moment
- `strain_vs_time_failure_6_5Nm.png`: Dynamic strain vs. time at high load
- `sensitivity_bar.png`: Parameter sensitivity analysis
- `spinal_model_analysis.png`: Comprehensive multi-panel analysis

### Validation

```bash
python validate_spinal_model_corrected.py
```

## File Structure

```
cord-avulsion-model/
├── README.md                           # This file
├── requirements.txt                     # Python dependencies
├── spinal_model.py                     # Core model implementation
├── analyze_model.py                    # Figure generation and analysis
├── validate_spinal_model_corrected.py  # Corrected validation script
├── validate_spinal_model.py            # Original validation script
├── test_spinal_model.py               # Unit tests
├── VALIDATION_SUMMARY.md              # Detailed validation results
├── figures/                           # Generated figures
│   ├── strain_vs_moment.png
│   ├── strain_vs_time_failure_6_5Nm.png
│   ├── sensitivity_bar.png
│   └── spinal_model_analysis.png
└── examples/                          # Example scripts
```

## Model Parameters

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| K_theta | 12 | Nm/rad | Disc-ligament stiffness |
| C_theta | 0.12 | Nms/rad | Damping coefficient |
| E_c | 0.3e6 | Pa | Cord elastic modulus |
| A_c | 3.8e-6 | m² | Cord cross-sectional area |
| L_0 | 0.102 | m | Cord rest length |
| r_c | 0.019 | m | Cord moment arm |
| I_theta | 4.8e-4 | kg⋅m² | Rotational inertia |
| ε_fail | 0.095 | - | Failure strain |

## Key Results

### Figure 1: Quasi-static Response
Shows the linear relationship between applied moment and cord strain, with the biological failure range (ε = 0.10 ± 0.02) indicated.

### Figure 2: Dynamic Response at High Load
Demonstrates rapid failure onset (20.6 ms) under 6.5 Nm loading, highlighting the time-critical nature of avulsion injuries.

### Figure 3: Parameter Sensitivity
Reveals that rotational stiffness (Kθ), moment arm (r_c), and damping (Cθ) have the greatest influence on failure risk and timing.

## Validation

The model has been validated against:
- ✅ Quasi-static response accuracy
- ✅ Strain calculation correctness  
- ✅ Energy conservation (undamped case)
- ✅ Natural frequency (25.2 Hz)
- ✅ Damping ratio (0.790)

## Citation

If you use this model in your research, please cite:

```
[Your paper citation here]
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here] 
=======
# cord-avulsion-model
A biomechanical model for thoracic spinal cord avulsion during high-speed hyperflexion
>>>>>>> aacc33938649a4201058c9bd2108ed76f0b5cebe
