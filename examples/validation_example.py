"""
Validation example for the Spinal Cord Avulsion Model.

This script demonstrates how to:
1. Validate the model's physical consistency
2. Check derived parameters
3. Verify energy conservation
4. Test strain calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from spinal_model import SpinalCordModel

def validate_physics():
    """Run comprehensive physics validation."""
    
    print("=== Model Physics Validation ===\n")
    
    model = SpinalCordModel()
    
    # 1. Check derived parameters
    print("1. Derived Parameters:")
    k_c_expected = model.E_c * model.A_c / model.L_0
    K_eff_expected = model.K_theta + k_c_expected * model.r_c**2
    
    print(f"   Cord stiffness (k_c): {model.k_c:.2f} N/m (expected: {k_c_expected:.2f})")
    print(f"   Effective stiffness (K_eff): {model.K_eff:.2f} Nm/rad (expected: {K_eff_expected:.2f})")
    
    # 2. Check quasi-static response
    print("\n2. Quasi-static Response:")
    M_test = 0.5
    phi_expected = M_test / model.K_eff
    t, phi, phi_dot, strain = model.simulate(M_test, t_final=0.5)
    phi_final = phi[-1]
    
    print(f"   Applied moment: {M_test} Nm")
    print(f"   Expected deflection: {np.rad2deg(phi_expected):.3f} degrees")
    print(f"   Final deflection: {np.rad2deg(phi_final):.3f} degrees")
    print(f"   Error: {abs(phi_final - phi_expected):.6f} rad")
    
    # 3. Check strain calculation
    print("\n3. Strain Calculation:")
    test_angle = 0.1
    expected_strain = (model.r_c * test_angle) / model.L_0
    computed_strain = model.calculate_strain(test_angle)
    
    print(f"   Test angle: {np.rad2deg(test_angle):.1f} degrees")
    print(f"   Expected strain: {expected_strain:.6f}")
    print(f"   Computed strain: {computed_strain:.6f}")
    print(f"   Error: {abs(computed_strain - expected_strain):.2e}")
    
    # 4. Energy conservation check
    print("\n4. Energy Conservation:")
    model_undamped = SpinalCordModel()
    model_undamped.C_theta = 0
    y0 = [1e-4, 0]
    t = np.linspace(0, 0.1, 1000)
    
    sol = odeint(model_undamped.system_dynamics, y0, t, args=(0,))
    phi = sol[:, 0]
    phi_dot = sol[:, 1]
    
    KE = 0.5 * model_undamped.I_theta * phi_dot**2
    PE = 0.5 * model_undamped.K_eff * phi**2
    E_total = KE + PE
    
    mean_E = np.mean(E_total)
    E_variation = (np.max(E_total) - np.min(E_total)) / mean_E if mean_E > 0 else 0
    
    print(f"   Mean total energy: {mean_E:.2e} J")
    print(f"   Energy variation: {E_variation:.2e}")
    print(f"   Conservation quality: {'Excellent' if E_variation < 1e-3 else 'Good' if E_variation < 1e-2 else 'Poor'}")
    
    # 5. System dynamics check
    print("\n5. System Dynamics:")
    omega_n = np.sqrt(model.K_eff / model.I_theta)
    f_n = omega_n / (2 * np.pi)
    zeta = model.C_theta / (2 * np.sqrt(model.K_eff * model.I_theta))
    
    print(f"   Natural frequency: {f_n:.1f} Hz")
    print(f"   Damping ratio: {zeta:.3f}")
    print(f"   System type: {'Underdamped' if zeta < 1 else 'Overdamped' if zeta > 1 else 'Critically damped'}")
    
    return True

def plot_validation_results():
    """Generate validation plots."""
    
    model = SpinalCordModel()
    
    # Energy conservation plot
    model_undamped = SpinalCordModel()
    model_undamped.C_theta = 0
    y0 = [1e-3, 0]
    t = np.linspace(0, 0.1, 1000)
    
    sol = odeint(model_undamped.system_dynamics, y0, t, args=(0,))
    phi = sol[:, 0]
    phi_dot = sol[:, 1]
    
    KE = 0.5 * model_undamped.I_theta * phi_dot**2
    PE = 0.5 * model_undamped.K_eff * phi**2
    E_total = KE + PE
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Energy plot
    ax1.plot(t*1000, KE, 'r-', label='Kinetic Energy', linewidth=2)
    ax1.plot(t*1000, PE, 'g-', label='Potential Energy', linewidth=2)
    ax1.plot(t*1000, E_total, 'b-', label='Total Energy', linewidth=2)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Energy (J)')
    ax1.set_title('Energy Conservation (Undamped)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Strain vs angle relationship
    angles = np.linspace(0, np.deg2rad(30), 100)
    strains = model.calculate_strain(angles)
    
    ax2.plot(np.rad2deg(angles), strains, 'b-', linewidth=2)
    ax2.axhline(model.eps_fail, color='r', linestyle='--', alpha=0.7, label=f'Failure threshold ({model.eps_fail})')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Strain')
    ax2.set_title('Strain vs Angle Relationship')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run validation example."""
    
    # Run validation
    validate_physics()
    
    # Generate validation plots
    print("\nGenerating validation plots...")
    plot_validation_results()
    
    print("\nValidation example completed!")
    print("Check 'examples/validation_results.png' for the validation plots.")

if __name__ == "__main__":
    main() 