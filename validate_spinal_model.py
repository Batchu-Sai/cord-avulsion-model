"""
Validation script for spinal cord model
"""
from spinal_model import SpinalCordModel
import numpy as np

def validate_physics():
    """Validate physical consistency of the model"""
    model = SpinalCordModel()
    
    # 1. Check derived parameters
    print("Checking derived parameters...")
    k_c = model.k_c
    K_eff = model.K_eff
    print(f"Cord stiffness (k_c): {k_c:.2f} N/m")
    print(f"Effective stiffness (K_eff): {K_eff:.2f} Nm/rad")
    
    # 2. Check quasi-static response
    M_test = 0.5  # Small test moment
    phi_expected = M_test / K_eff  # Static deflection
    
    t, phi, phi_dot, strain = model.simulate(M_test, t_final=0.5)  # Long enough to reach steady state
    phi_final = phi[-1]
    
    print("\nChecking quasi-static response...")
    print(f"Expected static deflection: {np.rad2deg(phi_expected):.2f} degrees")
    print(f"Simulated final deflection: {np.rad2deg(phi_final):.2f} degrees")
    
    # 3. Check strain relationship
    test_angle = 0.1  # Test angle in radians
    expected_strain = (model.r_c * test_angle) / model.L_0
    computed_strain = model.calculate_strain(test_angle)
    
    print("\nChecking strain calculation...")
    print(f"Expected strain at {np.rad2deg(test_angle):.1f} degrees: {expected_strain:.4f}")
    print(f"Computed strain at {np.rad2deg(test_angle):.1f} degrees: {computed_strain:.4f}")
    
    # 4. Check energy conservation (no damping, no input)
    model_undamped = SpinalCordModel()
    model_undamped.C_theta = 0  # Remove damping
    
    t, phi, phi_dot, _ = model_undamped.simulate(0, t_final=0.1)  # No external moment
    
    # Calculate total energy (kinetic + potential)
    KE = 0.5 * model_undamped.I_theta * phi_dot**2
    PE = 0.5 * model_undamped.K_eff * phi**2
    E_total = KE + PE
    
    E_variation = (np.max(E_total) - np.min(E_total)) / np.mean(E_total)
    print("\nChecking energy conservation (undamped case)...")
    print(f"Relative energy variation: {E_variation:.2e}")
    
    return all([
        abs(k_c - 52.16) < 0.1,  # Expected cord stiffness
        abs(K_eff - 5.22) < 0.1,  # Expected effective stiffness
        abs(phi_final - phi_expected) < 0.01,  # Static response
        abs(computed_strain - expected_strain) < 1e-10,  # Strain calculation
        E_variation < 1e-4  # Energy conservation
    ])

if __name__ == "__main__":
    print("Running physical validation of spinal cord model...\n")
    if validate_physics():
        print("\nValidation PASSED - Model is physically consistent!")
    else:
        print("\nValidation FAILED - Check physics implementation!")