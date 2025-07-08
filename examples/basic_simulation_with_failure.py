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


def plot_failure_figure(K_theta=3.0, M_0=2.0, ε_fail=0.095, t_final=0.1, n_points=1000):
    """
    Plot dynamic strain response with failure annotation for publication.
    Parameters:
        K_theta (float): Reduced disc-ligament stiffness [Nm/rad]
        M_0 (float): Step input moment [Nm]
        ε_fail (float): Biological failure threshold strain
    """
    model = SpinalCordModel()
    model.K_theta = K_theta
    model.K_eff = model.K_theta + model.k_c * model.r_c**2

    t, phi, phi_dot, strain = model.simulate(M_0=M_0, t_final=t_final, n_points=n_points)
    max_strain = np.max(strain)
    fail_idx = np.argmax(strain >= ε_fail)
    failure_detected = np.any(strain >= ε_fail)
    t_fail = t[fail_idx] if failure_detected else None

    plt.figure(figsize=(8, 6))
    plt.plot(t * 1000, strain, label=f"Cord Strain (ε) at M₀ = {M_0} Nm", linewidth=2)
    plt.axhline(y=ε_fail, color='black', linestyle='--', linewidth=2, label=f"Failure Threshold (ε = {ε_fail:.3f})")

    if failure_detected:
        plt.plot(t_fail * 1000, strain[fail_idx], 'ro', label=f"Failure at {t_fail*1000:.1f} ms")
        plt.annotate("Failure", xy=(t_fail * 1000, strain[fail_idx]),
                     xytext=(t_fail * 1000 + 5, strain[fail_idx] + 0.002),
                     arrowprops=dict(facecolor='red', arrowstyle="->"))

    plt.ylim(0, max_strain + 0.01)
    plt.xlabel("Time (ms)")
    plt.ylabel("Cord Strain")
    plt.title(f"Dynamic Cord Strain — Failure Confirmed\nMax Strain = {max_strain:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_basic_simulation()
    plot_failure_figure()