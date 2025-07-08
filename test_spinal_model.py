"""
Unit tests for the Spinal Cord Avulsion Model.

This module contains comprehensive tests for:
- Model initialization and parameter validation
- Strain calculations
- Simulation functionality
- Failure detection
- Energy conservation
"""

import unittest
import numpy as np
from spinal_model import SpinalCordModel

class TestSpinalCordModel(unittest.TestCase):
    """Test cases for SpinalCordModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SpinalCordModel()
    
    def test_model_initialization(self):
        """Test model initialization and default parameters."""
        model = SpinalCordModel()
        
        # Check that all parameters are positive
        self.assertGreater(model.K_theta, 0)
        self.assertGreater(model.C_theta, 0)
        self.assertGreater(model.E_c, 0)
        self.assertGreater(model.A_c, 0)
        self.assertGreater(model.L_0, 0)
        self.assertGreater(model.r_c, 0)
        self.assertGreater(model.I_theta, 0)
        self.assertGreater(model.eps_fail, 0)
        
        # Check derived parameters
        self.assertGreater(model.k_c, 0)
        self.assertGreater(model.K_eff, 0)
        
        # Check physical consistency
        self.assertGreater(model.K_eff, model.K_theta)
    
    def test_strain_calculation(self):
        """Test strain calculation for various angles."""
        model = SpinalCordModel()
        
        # Test zero angle
        strain = model.calculate_strain(0)
        self.assertEqual(strain, 0)
        
        # Test small angle
        angle = 0.1  # rad
        expected_strain = (model.r_c * angle) / model.L_0
        strain = model.calculate_strain(angle)
        self.assertAlmostEqual(strain, expected_strain, places=10)
        
        # Test larger angle
        angle = np.pi/6  # 30 degrees
        expected_strain = (model.r_c * angle) / model.L_0
        strain = model.calculate_strain(angle)
        self.assertAlmostEqual(strain, expected_strain, places=10)
        
        # Test array input
        angles = np.array([0, 0.1, np.pi/6])
        strains = model.calculate_strain(angles)
        self.assertEqual(len(strains), len(angles))
        self.assertEqual(strains[0], 0)
    
    def test_simulation_output(self):
        """Test simulation output format and properties."""
        model = SpinalCordModel()
        M_0 = 2.0
        t_final = 0.1
        
        t, phi, phi_dot, strain = model.simulate(M_0, t_final)
        
        # Check output types and shapes
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(phi, np.ndarray)
        self.assertIsInstance(phi_dot, np.ndarray)
        self.assertIsInstance(strain, np.ndarray)
        
        # Check array lengths
        self.assertEqual(len(t), len(phi))
        self.assertEqual(len(t), len(phi_dot))
        self.assertEqual(len(t), len(strain))
        
        # Check time array properties
        self.assertGreater(len(t), 1)
        self.assertEqual(t[0], 0)
        self.assertAlmostEqual(t[-1], t_final, places=3)
        
        # Check physical constraints
        self.assertTrue(np.all(strain >= 0))  # Strain should be non-negative
        self.assertTrue(np.all(phi >= 0))     # Angle should be non-negative for positive moment
    
    def test_failure_detection(self):
        """Test failure detection functionality."""
        model = SpinalCordModel()
        
        # Test with moment that causes failure
        M_high = 6.5
        t, phi, phi_dot, strain = model.simulate(M_high, t_final=0.1)
        
        # Check if failure is detected
        max_strain = np.max(strain)
        if max_strain >= model.eps_fail:
            # Find failure time
            fail_idx = np.argmax(strain >= model.eps_fail)
            fail_time = t[fail_idx]
            self.assertGreater(fail_time, 0)
            self.assertLess(fail_time, 0.1)
        else:
            # If no failure, max strain should be less than failure threshold
            self.assertLess(max_strain, model.eps_fail)
    
    def test_energy_conservation(self):
        """Test energy conservation in undamped case."""
        model = SpinalCordModel()
        model.C_theta = 0  # Remove damping
        
        # Small initial condition
        y0 = [1e-4, 0]
        t = np.linspace(0, 0.1, 1000)
        
        from scipy.integrate import odeint
        sol = odeint(model.system_dynamics, y0, t, args=(0,))
        phi = sol[:, 0]
        phi_dot = sol[:, 1]
        
        # Calculate energies
        KE = 0.5 * model.I_theta * phi_dot**2
        PE = 0.5 * model.K_eff * phi**2
        E_total = KE + PE
        
        # Check energy conservation
        mean_E = np.mean(E_total)
        if mean_E > 0:
            E_variation = (np.max(E_total) - np.min(E_total)) / mean_E
            self.assertLess(E_variation, 1e-2)  # Energy should be conserved within 1%
    
    def test_quasi_static_response(self):
        """Test quasi-static response accuracy."""
        model = SpinalCordModel()
        M_test = 0.5
        
        # Expected static deflection
        phi_expected = M_test / model.K_eff
        
        # Simulate and check final deflection
        t, phi, phi_dot, strain = model.simulate(M_test, t_final=0.5)
        phi_final = phi[-1]
        
        # Check that final deflection is close to expected
        error = abs(phi_final - phi_expected)
        self.assertLess(error, 0.01)  # Error should be less than 1%
    
    def test_parameter_validation(self):
        """Test parameter validation and bounds."""
        model = SpinalCordModel()
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            model.K_theta = -1
        
        with self.assertRaises(ValueError):
            model.E_c = 0
        
        with self.assertRaises(ValueError):
            model.eps_fail = 1.1  # Should be between 0 and 1
    
    def test_system_dynamics(self):
        """Test system dynamics function."""
        model = SpinalCordModel()
        
        # Test state vector
        y = [0.1, 0.5]  # [phi, phi_dot]
        M = 2.0
        
        # Get derivatives
        dy_dt = model.system_dynamics(y, 0, M)
        
        # Check output format
        self.assertEqual(len(dy_dt), 2)
        self.assertIsInstance(dy_dt[0], float)
        self.assertIsInstance(dy_dt[1], float)
    
    def test_failure_moment_calculation(self):
        """Test failure moment calculation."""
        model = SpinalCordModel()
        
        # Calculate failure moment
        M_fail = model.eps_fail * model.L_0 / model.r_c * model.K_eff
        
        # Verify by simulation
        t, phi, phi_dot, strain = model.simulate(M_fail, t_final=0.1)
        max_strain = np.max(strain)
        
        # Should be very close to failure threshold
        self.assertAlmostEqual(max_strain, model.eps_fail, places=3)

def run_tests():
    """Run all tests and print summary."""
    print("Running Spinal Cord Model Tests...\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpinalCordModel)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("  Status: ALL TESTS PASSED ✅")
    else:
        print("  Status: SOME TESTS FAILED ❌")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()