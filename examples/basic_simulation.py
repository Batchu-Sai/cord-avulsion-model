"""
Basic example of using the spinal cord model
"""

from spinal_model import SpinalCordModel
import numpy as np

def run_basic_simulation():
    # Create model
    model = SpinalCordModel()
    
    # Print model parameters
    print("Model Parameters:")
    print(f"Cord stiffness: {model.k_c:.2f} N/m")
    print(f"Effective stiffness: {model.K_eff:.2f} Nm/rad")
    
    # Run simulations at different moments
    moments = [0.4, 0.8, 1.2]
    
    for M_0 in moments:
        print(f"\nSimulating with moment = {M_0} Nm")
        t, phi, phi_dot, strain = model.simulate(M_0)
        
        # Print maximum values
        max_angle = np.max(np.abs(phi))
        max_strain = np.max(np.abs(strain))
        
        print(f"Maximum angle: {np.rad2deg(max_angle):.2f} degrees")
        print(f"Maximum strain: {max_strain:.4f}")
        print(f"Failure strain: {model.eps_fail}")
        print(f"Failure occurred: {max_strain > model.eps_fail}")
        
        # Plot response
        model.plot_step_response(M_0)

if __name__ == "__main__":
    run_basic_simulation()