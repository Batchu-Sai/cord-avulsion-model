"""
Analysis script for spinal cord model parameters and validation
"""
import numpy as np
import matplotlib.pyplot as plt
from spinal_model import SpinalCordModel
import os

def analyze_parameters():
    """Analyze and display model parameters"""
    model = SpinalCordModel()
    
    print("=== SPINAL CORD MODEL PARAMETER ANALYSIS ===\n")
    
    print("Physical Parameters:")
    print(f"Disc-ligament stiffness (K_theta): {model.K_theta} Nm/rad")
    print(f"Damping coefficient (C_theta): {model.C_theta} Nms/rad")
    print(f"Cord elastic modulus (E_c): {model.E_c:.1e} Pa")
    print(f"Cord cross-sectional area (A_c): {model.A_c:.1e} m²")
    print(f"Cord rest length (L_0): {model.L_0} m")
    print(f"Cord moment arm (r_c): {model.r_c} m")
    print(f"Rotational inertia (I_theta): {model.I_theta:.1e} kg⋅m²")
    print(f"Failure strain (eps_fail): {model.eps_fail}")
    
    print("\nDerived Parameters:")
    print(f"Cord stiffness (k_c = E_c * A_c / L_0): {model.k_c:.2f} N/m")
    print(f"Effective stiffness (K_eff = K_theta + k_c * r_c²): {model.K_eff:.2f} Nm/rad")
    
    # Calculate what the validation script expects
    print("\nValidation Script Expected Values:")
    print(f"Expected k_c: 52.16 N/m")
    print(f"Expected K_eff: 5.22 Nm/rad")
    
    print("\nDiscrepancy Analysis:")
    print(f"k_c ratio (actual/expected): {model.k_c/52.16:.3f}")
    print(f"K_eff ratio (actual/expected): {model.K_eff/5.22:.3f}")
    
    return model

def create_comprehensive_plots():
    """Create comprehensive plots of model behavior"""
    model = SpinalCordModel()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Step response for different moments
    ax1 = plt.subplot(3, 2, 1)
    moments = [0.2, 0.5, 0.8, 1.2]
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, M in enumerate(moments):
        t, phi, _, _ = model.simulate(M, t_final=0.1)
        ax1.plot(t*1000, np.rad2deg(phi), color=colors[i], 
                label=f'M = {M} Nm', linewidth=2)
    
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Angle [degrees]')
    ax1.set_title('Step Response - Different Moments')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Strain vs Angle relationship
    ax2 = plt.subplot(3, 2, 2)
    angles = np.linspace(0, np.deg2rad(30), 100)
    strains = model.calculate_strain(angles)
    
    ax2.plot(np.rad2deg(angles), strains, 'b-', linewidth=2)
    ax2.axhline(y=model.eps_fail, color='r', linestyle='--', 
               label=f'Failure Threshold ({model.eps_fail})')
    ax2.set_xlabel('Angle [degrees]')
    ax2.set_ylabel('Strain [-]')
    ax2.set_title('Strain vs Angle Relationship')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Frequency response (Bode plot approximation)
    ax3 = plt.subplot(3, 2, 3)
    frequencies = np.logspace(0, 3, 100)
    omega = 2 * np.pi * frequencies
    
    # Transfer function magnitude
    K_eff = model.K_eff
    I_theta = model.I_theta
    C_theta = model.C_theta
    
    # Magnitude of transfer function G(s) = 1/(I_theta*s² + C_theta*s + K_eff)
    magnitude = 1 / np.sqrt((K_eff - I_theta * omega**2)**2 + (C_theta * omega)**2)
    
    ax3.semilogx(frequencies, 20 * np.log10(magnitude), 'b-', linewidth=2)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Magnitude [dB]')
    ax3.set_title('Frequency Response (Magnitude)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase response
    ax4 = plt.subplot(3, 2, 4)
    phase = -np.arctan2(C_theta * omega, K_eff - I_theta * omega**2)
    ax4.semilogx(frequencies, np.rad2deg(phase), 'b-', linewidth=2)
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Phase [degrees]')
    ax4.set_title('Frequency Response (Phase)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy analysis
    ax5 = plt.subplot(3, 2, 5)
    # Run undamped simulation
    model_undamped = SpinalCordModel()
    model_undamped.C_theta = 0
    y0 = [1e-3, 0]  # Small initial angle
    t = np.linspace(0, 0.1, 1000)
    
    from scipy.integrate import odeint
    sol = odeint(model_undamped.system_dynamics, y0, t, args=(0,))
    phi = sol[:, 0]
    phi_dot = sol[:, 1]
    
    KE = 0.5 * model_undamped.I_theta * phi_dot**2
    PE = 0.5 * model_undamped.K_eff * phi**2
    E_total = KE + PE
    
    ax5.plot(t*1000, KE, 'r-', label='Kinetic Energy', linewidth=2)
    ax5.plot(t*1000, PE, 'g-', label='Potential Energy', linewidth=2)
    ax5.plot(t*1000, E_total, 'b-', label='Total Energy', linewidth=2)
    ax5.set_xlabel('Time [ms]')
    ax5.set_ylabel('Energy [J]')
    ax5.set_title('Energy Conservation (Undamped)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Parameter sensitivity
    ax6 = plt.subplot(3, 2, 6)
    # Vary cord elastic modulus
    E_c_values = np.linspace(0.1e6, 0.5e6, 50)
    k_c_values = (E_c_values * model.A_c) / model.L_0
    K_eff_values = model.K_theta + k_c_values * model.r_c**2
    
    ax6.plot(E_c_values/1e6, K_eff_values, 'b-', linewidth=2)
    ax6.set_xlabel('Cord Elastic Modulus [MPa]')
    ax6.set_ylabel('Effective Stiffness [Nm/rad]')
    ax6.set_title('Sensitivity: K_eff vs E_c')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spinal_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def run_validation_analysis():
    """Run detailed validation analysis"""
    model = SpinalCordModel()
    
    print("\n=== DETAILED VALIDATION ANALYSIS ===\n")
    
    # Test 1: Quasi-static response
    M_test = 0.5
    phi_expected = M_test / model.K_eff
    t, phi, _, _ = model.simulate(M_test, t_final=0.5)
    phi_final = phi[-1]
    
    print(f"Test 1 - Quasi-static response:")
    print(f"  Applied moment: {M_test} Nm")
    print(f"  Expected deflection: {np.rad2deg(phi_expected):.3f} degrees")
    print(f"  Final deflection: {np.rad2deg(phi_final):.3f} degrees")
    print(f"  Error: {abs(phi_final - phi_expected):.6f} rad")
    
    # Test 2: Strain calculation
    test_angle = 0.1
    expected_strain = (model.r_c * test_angle) / model.L_0
    computed_strain = model.calculate_strain(test_angle)
    
    print(f"\nTest 2 - Strain calculation:")
    print(f"  Test angle: {np.rad2deg(test_angle):.1f} degrees")
    print(f"  Expected strain: {expected_strain:.6f}")
    print(f"  Computed strain: {computed_strain:.6f}")
    print(f"  Error: {abs(computed_strain - expected_strain):.2e}")
    
    # Test 3: Energy conservation
    model_undamped = SpinalCordModel()
    model_undamped.C_theta = 0
    y0 = [1e-4, 0]
    t = np.linspace(0, 0.1, 1000)
    
    from scipy.integrate import odeint
    sol = odeint(model_undamped.system_dynamics, y0, t, args=(0,))
    phi = sol[:, 0]
    phi_dot = sol[:, 1]
    
    KE = 0.5 * model_undamped.I_theta * phi_dot**2
    PE = 0.5 * model_undamped.K_eff * phi**2
    E_total = KE + PE
    
    mean_E = np.mean(E_total)
    E_variation = (np.max(E_total) - np.min(E_total)) / mean_E if mean_E > 0 else 0
    
    print(f"\nTest 3 - Energy conservation:")
    print(f"  Mean total energy: {mean_E:.2e} J")
    print(f"  Energy variation: {E_variation:.2e}")
    print(f"  Conservation quality: {'Excellent' if E_variation < 1e-3 else 'Good' if E_variation < 1e-2 else 'Poor'}")

# --- New Figure Functions ---
def plot_strain_vs_moment():
    """Publication-style: quasi-static cord strain vs. applied moment with biological failure range (0.10±0.02)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18, 'legend.fontsize': 15, 'xtick.labelsize': 15, 'ytick.labelsize': 15})
    model = SpinalCordModel()
    moments = np.linspace(0, 7.0, 300)
    strains = moments / model.K_eff * model.r_c / model.L_0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(moments, strains, color='orange', lw=2.5, label=r'Quasi-static response')
    ax.fill_between(moments, 0.08, 0.12, color='gray', alpha=0.25, label=r'Failure range ($0.10 \pm 0.02$)')
    ax.set_xlabel(r'Applied Moment $M_0$ (Nm)')
    ax.set_ylabel(r'Spinal Cord Strain $\varepsilon$')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 0.14)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.18, linestyle='--')
    plt.tight_layout(pad=1.5)
    plt.savefig('strain_vs_moment.png', dpi=400, bbox_inches='tight')
    plt.show()
    os.system('open strain_vs_moment.png')

def plot_strain_vs_time_with_failure(M0=2.0):
    """Ultra-clean, publication-style plot: dynamic cord strain vs. time at M0, with only a red failure dot and label."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 17, 'axes.titlesize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
    model = SpinalCordModel()
    t, _, _, strain = model.simulate(M0, t_final=0.1)
    eps_fail = model.eps_fail
    fail_idx = np.argmax(strain >= eps_fail)
    fail_time = t[fail_idx] * 1000 if np.any(strain >= eps_fail) else None
    fig, ax = plt.subplots(figsize=(7, 4))
    # Strain curve
    ax.plot(t*1000, strain, color='orange', lw=2, zorder=1)
    # Threshold line
    ax.axhline(eps_fail, color='k', ls='--', lw=2, zorder=0)
    # Failure dot
    if fail_time is not None:
        ax.plot(t[fail_idx]*1000, strain[fail_idx], 'o', color='red', markersize=10, zorder=10)
        ax.text(t[fail_idx]*1000+2, strain[fail_idx]+0.003, 'Failure', color='red', fontsize=15, va='bottom', ha='left', weight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cord Strain')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(0.13, 1.05*max(strain)))
    ax.grid(True, alpha=0.13, linestyle='--')
    plt.tight_layout(pad=1.2)
    plt.savefig('strain_vs_time_failure.png', dpi=400, bbox_inches='tight')
    plt.show()
    os.system('open strain_vs_time_failure.png')

def plot_sensitivity_bar():
    """
    Sensitivity analysis using a fixed input moment (6.5 Nm).
    For each parameter, sweep ±50%, run the simulation, and record:
      - max strain
      - time to failure (if failure occurs)
      - whether failure occurred
    Plots relative changes for each parameter.
    """
    import matplotlib.pyplot as plt
    base = SpinalCordModel()
    params = ['Kθ', 'k_c', 'r_c', 'Cθ', 'εfail']
    base_values = [base.K_theta, base.k_c, base.r_c, base.C_theta, base.eps_fail]
    sweep_fracs = [0.5, 1.5]
    M_fixed = 6.5
    t, _, _, strain = base.simulate(M_fixed, t_final=0.1)
    max_strain_base = np.max(strain)
    fail_idx_base = np.argmax(strain >= base.eps_fail)
    if np.any(strain >= base.eps_fail):
        time_to_fail_base = t[fail_idx_base]*1000
        failure_base = True
    else:
        time_to_fail_base = float('nan')
        failure_base = False
    rel_max_strain = []
    rel_time_to_fail = []
    failure_flags = []
    for i, (pname, pval) in enumerate(zip(params, base_values)):
        max_strain_sweep = []
        time_to_fail_sweep = []
        failure_sweep = []
        for frac in sweep_fracs:
            model = SpinalCordModel()
            if pname == 'Kθ':
                model.K_theta = pval * frac
                model.K_eff = model.K_theta + model.k_c * model.r_c**2
            elif pname == 'k_c':
                model.E_c = base.E_c * frac
                model.k_c = (model.E_c * model.A_c) / model.L_0
                model.K_eff = model.K_theta + model.k_c * model.r_c**2
            elif pname == 'r_c':
                model.r_c = pval * frac
                model.K_eff = model.K_theta + model.k_c * model.r_c**2
            elif pname == 'Cθ':
                model.C_theta = pval * frac
            elif pname == 'εfail':
                model.eps_fail = pval * frac
            t, _, _, strain = model.simulate(M_fixed, t_final=0.1)
            max_strain = np.max(strain)
            fail_idx = np.argmax(strain >= model.eps_fail)
            if np.any(strain >= model.eps_fail):
                time_to_fail = t[fail_idx]*1000
                failure = True
            else:
                time_to_fail = float('nan')
                failure = False
            max_strain_sweep.append(max_strain)
            time_to_fail_sweep.append(time_to_fail)
            failure_sweep.append(failure)
        rel_max_strain.append([(v - max_strain_base)/max_strain_base for v in max_strain_sweep])
        rel_time_to_fail.append([(v - time_to_fail_base)/time_to_fail_base if failure_base and f else float('nan') for v, f in zip(time_to_fail_sweep, failure_sweep)])
        failure_flags.append(failure_sweep)
    rel_max_strain = np.array(rel_max_strain)
    rel_time_to_fail = np.array(rel_time_to_fail)
    failure_flags = np.array(failure_flags)
    width = 0.25
    x = np.arange(len(params))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for j, frac in enumerate(sweep_fracs):
        bars = ax[0].bar(x + (j-0.5)*width, rel_max_strain[:,j], width, label=f'{int(frac*100)}%')
        for i, bar in enumerate(bars):
            if not failure_flags[i, j]:
                ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), '×', ha='center', va='bottom', color='red', fontsize=16)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(params, fontsize=14)
    ax[0].set_ylabel('Relative Change in Max Strain', fontsize=13)
    ax[0].set_title('Max Strain Sensitivity', fontsize=14)
    ax[0].legend(title='Parameter Value', fontsize=12)
    ax[0].grid(True, axis='y', alpha=0.18, linestyle='--')
    for j, frac in enumerate(sweep_fracs):
        bars = ax[1].bar(x + (j-0.5)*width, rel_time_to_fail[:,j], width, label=f'{int(frac*100)}%')
        for i, bar in enumerate(bars):
            if not failure_flags[i, j]:
                ax[1].text(bar.get_x() + bar.get_width()/2, 0, '×', ha='center', va='bottom', color='red', fontsize=16)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(params, fontsize=14)
    ax[1].set_ylabel('Relative Change in Time to Failure', fontsize=13)
    ax[1].set_title('Time to Failure Sensitivity', fontsize=14)
    ax[1].legend(title='Parameter Value', fontsize=12)
    ax[1].grid(True, axis='y', alpha=0.18, linestyle='--')
    plt.tight_layout()
    plt.savefig('sensitivity_bar.png', dpi=400, bbox_inches='tight')
    plt.show()
    os.system('open sensitivity_bar.png')

def plot_strain_vs_time_with_failure_high_moment():
    """Ultra-clean publication figure: dynamic cord strain vs. time at 6.5 Nm, no legend, minimal elements."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import os
    mpl.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'serif',
        'axes.linewidth': 1.1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.1,
        'ytick.major.width': 1.1,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
    })
    model = SpinalCordModel()
    M0 = 6.5
    t, _, _, strain = model.simulate(M0, t_final=0.1)
    eps_fail = model.eps_fail
    fail_idx = np.argmax(strain >= eps_fail)
    fail_time = t[fail_idx] * 1000 if np.any(strain >= eps_fail) else None
    fig, ax = plt.subplots(figsize=(7, 4))
    # Strain curve (no label)
    ax.plot(t*1000, strain, color='orange', lw=1.5, zorder=1)
    # Threshold line (no label)
    ax.axhline(eps_fail, color='k', ls='--', lw=1.5, zorder=0)
    # Failure dot and annotation
    if fail_time is not None:
        ax.plot(t[fail_idx]*1000, strain[fail_idx], 'o', color='red', markersize=8, zorder=10)
        ax.text(t[fail_idx]*1000, strain[fail_idx]-0.008, f'Failure at {fail_time:.1f} ms',
                color='gray', fontsize=12, fontfamily='serif', va='top', ha='center')
    ax.set_xlabel(r'Time (ms)')
    ax.set_ylabel(r'Cord Strain ($\varepsilon$)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(0.13, 1.05*np.max(strain)))
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7))
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.7)
    plt.tight_layout(pad=1.2)
    plt.savefig('strain_vs_time_failure_6_5Nm.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()
    os.system('open strain_vs_time_failure_6_5Nm.png')

# --- End New Figure Functions ---

if __name__ == "__main__":
    # Run parameter analysis
    model = analyze_parameters()
    
    # Run validation analysis
    run_validation_analysis()
    
    # Create comprehensive plots
    print("\nGenerating comprehensive plots...")
    create_comprehensive_plots()
    
    print("\nGenerating requested figures...")
    plot_strain_vs_moment()
    plot_strain_vs_time_with_failure()
    plot_sensitivity_bar()
    plot_strain_vs_time_with_failure_high_moment()
    print("\nAll requested figures generated!") 