"""
Validation script for spinal cord model
"""
from spinal_model import SpinalCordModel
import numpy as np

def validate_physics():
    model = SpinalCordModel()
    
    # 1. Check derived parameters
    print("Checking derived parameters...")
    k_c = model.k_c
    K_eff = model.K_eff
    print(f"Cord stiffness (k_c): {k_c:.2f} N/m")
    print(f"Effective stiffness (K_eff): {K_eff:.2f} Nm/rad")
    
    # 2. Check quasi-static response
    M_test = 0.5
    phi_expected = M_test / K_eff
    t, phi, phi_dot, strain = model.simulate(M_test, t_final=0.5)
    phi_final = phi[-1]
    
    print("\\nChecking quasi-static response...")
    print(f"Expected static deflection: {np.rad2deg(phi_expected):.2f} degrees")
    print(f"Simulated final deflection: {np.rad2deg(phi_final):.2f} degrees")
    
    # 3. Check strain relationship
    test_angle = 0.1
    expected_strain = (model.r_c * test_angle) / model.L_0
    computed_strain = model.calculate_strain(test_angle)
    
    print("\\nChecking strain calculation...")
    print(f"Expected strain at {np.rad2deg(test_angle):.1f} degrees: {expected_strain:.4f}")
    print(f"Computed strain at {np.rad2deg(test_angle):.1f} degrees: {computed_strain:.4f}")
    
    # 4. Energy conservation check (undamped, zero input, small initial condition)
    model_undamped = SpinalCordModel()
    model_undamped.C_theta = 0
    y0 = [1e-4, 0]  # Small angle perturbation
    t = np.linspace(0, 0.1, 1000)
    
    from scipy.integrate import odeint
    sol = odeint(model_undamped.system_dynamics, y0, t, args=(0,))
    phi = sol[:, 0]
    phi_dot = sol[:, 1]
    
    KE = 0.5 * model_undamped.I_theta * phi_dot**2
    PE = 0.5 * model_undamped.K_eff * phi**2
    E_total = KE + PE
    
    mean_E = np.mean(E_total)
    if mean_E > 0:
        E_variation = (np.max(E_total) - np.min(E_total)) / mean_E
    else:
        E_variation = 0.0
    
    print("\\nChecking energy conservation (undamped case)...")
    print(f"Relative energy variation: {E_variation:.2e}")
    
    return all([
        abs(k_c - 52.16) < 0.1,
        abs(K_eff - 5.22) < 0.1,
        abs(phi_final - phi_expected) < 0.01,
        abs(computed_strain - expected_strain) < 1e-10,
        E_variation < 2e-4
    ])

if __name__ == "__main__":
    print("Running physical validation of spinal cord model...\\n")
    if validate_physics():
        print("\\nValidation PASSED - Model is physically consistent!")
    else:
        print("\\nValidation FAILED - Check physics implementation!")