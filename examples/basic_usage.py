"""
Basic usage example for the Spinal Cord Avulsion Model.

This script demonstrates how to:
1. Create and configure the model
2. Run simulations
3. Analyze results
4. Generate basic plots
"""

import numpy as np
import matplotlib.pyplot as plt
from spinal_model import SpinalCordModel

def main():
    """Demonstrate basic model usage."""
    
    print("=== Spinal Cord Avulsion Model - Basic Usage ===\n")
    
    # 1. Create model instance
    model = SpinalCordModel()
    
    # 2. Display model parameters
    print("Model Parameters:")
    print(f"  Cord stiffness (k_c): {model.k_c:.2f} N/m")
    print(f"  Effective stiffness (K_eff): {model.K_eff:.2f} Nm/rad")
    print(f"  Failure strain (ε_fail): {model.eps_fail}")
    print(f"  Natural frequency: {np.sqrt(model.K_eff/model.I_theta)/(2*np.pi):.1f} Hz")
    print(f"  Damping ratio: {model.C_theta/(2*np.sqrt(model.K_eff*model.I_theta)):.3f}")
    
    # 3. Calculate failure moment
    M_fail = model.eps_fail * model.L_0 / model.r_c * model.K_eff
    print(f"\nFailure moment (quasi-static): {M_fail:.2f} Nm")
    
    # 4. Run simulation at different moments
    moments = [2.0, 4.0, 6.5]
    fig, axes = plt.subplots(1, len(moments), figsize=(15, 4))
    
    for i, M in enumerate(moments):
        # Run simulation
        t, phi, phi_dot, strain = model.simulate(M, t_final=0.1)
        
        # Check if failure occurred
        fail_idx = np.argmax(strain >= model.eps_fail)
        if np.any(strain >= model.eps_fail):
            fail_time = t[fail_idx] * 1000
            print(f"  M = {M} Nm: Failure at {fail_time:.1f} ms")
        else:
            print(f"  M = {M} Nm: No failure (max strain = {np.max(strain):.4f})")
        
        # Plot strain vs time
        axes[i].plot(t*1000, strain, 'b-', linewidth=2)
        axes[i].axhline(model.eps_fail, color='r', linestyle='--', alpha=0.7)
        axes[i].set_xlabel('Time (ms)')
        axes[i].set_ylabel('Strain')
        axes[i].set_title(f'M = {M} Nm')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/basic_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Parameter sensitivity example
    print("\nParameter Sensitivity Example:")
    base_strain = np.max(model.simulate(6.5, t_final=0.1)[3])
    
    # Test different cord elastic moduli
    E_c_values = [0.2e6, 0.3e6, 0.4e6]  # Pa
    strains = []
    
    for E_c in E_c_values:
        test_model = SpinalCordModel()
        test_model.E_c = E_c
        test_model.k_c = (test_model.E_c * test_model.A_c) / test_model.L_0
        test_model.K_eff = test_model.K_theta + test_model.k_c * test_model.r_c**2
        
        strain = np.max(test_model.simulate(6.5, t_final=0.1)[3])
        strains.append(strain)
        print(f"  E_c = {E_c/1e6:.1f} MPa: max strain = {strain:.4f}")
    
    print("\nBasic usage example completed!")
    print("Check 'examples/basic_simulation.png' for the generated plot.")

if __name__ == "__main__":
    main() 