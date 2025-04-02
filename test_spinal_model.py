"""
Tests for the spinal cord model
"""

import numpy as np
from spinal_model import SpinalCordModel

def test_parameters():
    """Verify that derived parameters match expected values."""
    model = SpinalCordModel()
    
    # Test cord stiffness calculation
    expected_k_c = (1.4e6 * 3.8e-6) / 0.102  # E_c * A_c / L_0
    assert np.isclose(model.k_c, expected_k_c, rtol=1e-10)
    print(f"Cord stiffness test passed: {model.k_c:.2f} N/m")
    
    # Test effective stiffness calculation
    expected_K_eff = 5.2 + expected_k_c * (0.019**2)
    assert np.isclose(model.K_eff, expected_K_eff, rtol=1e-10)
    print(f"Effective stiffness test passed: {model.K_eff:.2f} Nm/rad")

def test_strain_calculation():
    """Verify strain calculation for known angles."""
    model = SpinalCordModel()
    
    # Test at 0.1 radians
    strain_01 = model.calculate_strain(0.1)
    expected_strain = (0.019 * 0.1) / 0.102
    assert np.isclose(strain_01, expected_strain, rtol=1e-10)
    print(f"Strain calculation test passed: {strain_01:.4f} at 0.1 rad")

def test_dynamics():
    """Test the system dynamics function."""
    model = SpinalCordModel()
    
    # Test with zero moment and zero state
    state = [0, 0]
    result = model.system_dynamics(state, 0, 0)
    assert np.allclose(result, [0, 0], rtol=1e-10)
    print("Zero state dynamics test passed")
    
    # Test with unit moment
    state = [0, 0]
    result = model.system_dynamics(state, 0, 1.0)
    expected_accel = 1.0 / model.I_theta
    assert np.isclose(result[1], expected_accel, rtol=1e-10)
    print(f"Unit moment dynamics test passed: acceleration = {result[1]:.2f} rad/s²")

def test_simulation():
    """Test the simulation results."""
    model = SpinalCordModel()
    
    # Run short simulation
    t, phi, phi_dot, strain = model.simulate(M_0=0.8, t_final=0.01)
    
    # Check lengths
    assert len(t) == len(phi) == len(phi_dot) == len(strain)
    print(f"Simulation dimensions test passed: {len(t)} points")
    
    # Check initial conditions
    assert np.allclose([phi[0], phi_dot[0]], [0, 0], rtol=1e-10)
    print("Initial conditions test passed")
    
    # Check physical constraints
    max_strain = np.max(np.abs(strain))
    assert max_strain < model.eps_fail  # Should not fail at 0.8 Nm
    print(f"Physical constraints test passed: max strain = {max_strain:.4f}")

if __name__ == "__main__":
    print("Running spinal cord model tests...")
    print("\nParameter Tests:")
    test_parameters()
    
    print("\nStrain Calculation Tests:")
    test_strain_calculation()
    
    print("\nDynamics Tests:")
    test_dynamics()
    
    print("\nSimulation Tests:")
    test_simulation()
    print("\nAll tests passed successfully!")