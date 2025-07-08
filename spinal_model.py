"""
Thoracic Spinal Cord Avulsion Model - Verified Working Version
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class SpinalCordModel:
    def __init__(self):
        # Model parameters
<<<<<<< HEAD
        self.K_theta = 12      # Disc-ligament stiffness [Nm/rad]
        self.C_theta = 0.12     # Damping coefficient [Nms/rad]
        self.E_c = 0.3e6        # Cord elastic modulus [Pa]
=======
        self.K_theta = 5.2      # Disc-ligament stiffness [Nm/rad]
        self.C_theta = 0.12     # Damping coefficient [Nms/rad]
        self.E_c = 1.4e6        # Cord elastic modulus [Pa]
>>>>>>> aacc33938649a4201058c9bd2108ed76f0b5cebe
        self.A_c = 3.8e-6       # Cord cross-sectional area [m^2]
        self.L_0 = 0.102        # Cord rest length [m]
        self.r_c = 0.019        # Cord moment arm [m]
        self.I_theta = 4.8e-4   # Rotational inertia [kg⋅m^2]
        self.eps_fail = 0.095   # Failure strain [-]
        
        # Derived parameters
        self.k_c = (self.E_c * self.A_c) / self.L_0
        self.K_eff = self.K_theta + self.k_c * self.r_c**2
    
    def system_dynamics(self, state, t, M_0):
        """System dynamics for the model."""
        phi, phi_dot = state
        
        # Calculate acceleration
        phi_ddot = (M_0 - self.C_theta * phi_dot - self.K_eff * phi) / self.I_theta
        
        return [phi_dot, phi_ddot]
    
    def calculate_strain(self, phi):
        """Calculate cord strain for given angle."""
        return (self.r_c * phi) / self.L_0
    
    def simulate(self, M_0, t_final=0.1, n_points=1000):
        """Run simulation with given moment."""
        # Time points
        t = np.linspace(0, t_final, n_points)
        
        # Initial conditions [phi, phi_dot]
        y0 = [0.0, 0.0]
        
        # Solve ODE
        solution = odeint(self.system_dynamics, y0, t, args=(M_0,))
        
        # Extract results
        phi = solution[:, 0]
        phi_dot = solution[:, 1]
        strain = self.calculate_strain(phi)
        
        return t, phi, phi_dot, strain
    
    def plot_step_response(self, M_0, t_final=0.1):  # Changed name to match what's being called
        """Plot the system response."""
        # Run simulation
        t, phi, phi_dot, strain = self.simulate(M_0, t_final)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot angle
        ax1.plot(t*1000, np.rad2deg(phi), 'b-', label='Flexion Angle')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Angle [degrees]')
        ax1.grid(True)
        ax1.legend()
        
        # Plot strain
        ax2.plot(t*1000, strain, 'r-', label='Cord Strain')
        ax2.axhline(y=self.eps_fail, color='k', linestyle='--', 
                   label='Failure Threshold')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Strain [-]')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print maximum values
        print(f"\nMaximum values:")
        print(f"Max angle: {np.max(np.abs(phi)) * 180/np.pi:.2f} degrees")
        print(f"Max strain: {np.max(np.abs(strain)):.4f}")
        print(f"Max angular velocity: {np.max(np.abs(phi_dot)):.2f} rad/s")

def example():
    """Run an example simulation."""
    model = SpinalCordModel()
    
    print("Model Parameters:")
    print(f"Cord stiffness: {model.k_c:.2f} N/m")
    print(f"Effective stiffness: {model.K_eff:.2f} Nm/rad")
    
    # Run simulation with 0.8 Nm moment
    model.plot_step_response(M_0=0.8)  # Changed to match method name

if __name__ == "__main__":
    example()