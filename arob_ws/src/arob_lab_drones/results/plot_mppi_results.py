#!/usr/bin/env python3
"""
Plot MPPI flight results from CSV log (NO PANDAS VERSION)
Usage: python3 plot_mppi_results.py mppi_log_1234567890.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import csv
import glob

def load_csv_data(csv_path):
    """Load CSV data into dictionary of lists (no pandas)"""
    data = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Initialize lists for each column
        for fieldname in reader.fieldnames:
            data[fieldname] = []
        
        # Read all rows
        for row in reader:
            for key, value in row.items():
                try:
                    # Convert to float
                    data[key].append(float(value))
                except ValueError:
                    # Keep as string if conversion fails
                    data[key].append(value)
    
    # Convert lists to numpy arrays for easier math
    for key in data:
        data[key] = np.array(data[key])
    
    return data

def plot_mppi_results(csv_path):
    """Generate comprehensive plots from MPPI log CSV"""
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    # Load data
    data = load_csv_data(csv_path)
    print(f"Loaded {len(data['time'])} data points from {csv_path}")
    print(f"Flight duration: {data['time'][-1]:.2f} seconds")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ========== PLOT 1: Position Tracking (X, Y, Z) ==========
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(data['time'], data['pos_x'], 'b-', label='Actual X', linewidth=2)
    ax1.plot(data['time'], data['ref_px'], 'b--', label='Reference X', linewidth=1.5, alpha=0.7)
    ax1.plot(data['time'], data['pos_y'], 'r-', label='Actual Y', linewidth=2)
    ax1.plot(data['time'], data['ref_py'], 'r--', label='Reference Y', linewidth=1.5, alpha=0.7)
    ax1.plot(data['time'], data['pos_z'], 'g-', label='Actual Z', linewidth=2)
    ax1.plot(data['time'], data['ref_pz'], 'g--', label='Reference Z', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Position (m)', fontsize=10)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Position Tracking (X, Y, Z)', fontsize=11, fontweight='bold')
    
    # ========== PLOT 2: Position Error ==========
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(data['time'], data['err_pos'], 'k-', linewidth=2)
    ax2.fill_between(data['time'], 0, data['err_pos'], alpha=0.3)
    ax2.set_ylabel('Position Error (m)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Position Error (Mean: {np.mean(data["err_pos"]):.3f}m, Max: {np.max(data["err_pos"]):.3f}m)', 
                  fontsize=11, fontweight='bold')
    
    # ========== PLOT 3: Velocity Tracking (X, Y, Z) ==========
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(data['time'], data['vel_x'], 'b-', label='Actual vx', linewidth=2)
    ax3.plot(data['time'], data['ref_vx'], 'b--', label='Reference vx', linewidth=1.5, alpha=0.7)
    ax3.plot(data['time'], data['vel_y'], 'r-', label='Actual vy', linewidth=2)
    ax3.plot(data['time'], data['ref_vy'], 'r--', label='Reference vy', linewidth=1.5, alpha=0.7)
    ax3.plot(data['time'], data['vel_z'], 'g-', label='Actual vz', linewidth=2)
    ax3.plot(data['time'], data['ref_vz'], 'g--', label='Reference vz', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Velocity (m/s)', fontsize=10)
    ax3.legend(loc='best', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Velocity Tracking (X, Y, Z)', fontsize=11, fontweight='bold')
    
    # ========== PLOT 4: Velocity Error ==========
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(data['time'], data['err_vel'], 'k-', linewidth=2)
    ax4.fill_between(data['time'], 0, data['err_vel'], alpha=0.3)
    ax4.set_ylabel('Velocity Error (m/s)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f'Velocity Error (Mean: {np.mean(data["err_vel"]):.3f}m/s, Max: {np.max(data["err_vel"]):.3f}m/s)', 
                  fontsize=11, fontweight='bold')
    
    # ========== PLOT 5: Commands vs Actual Velocity ==========
    ax5 = plt.subplot(4, 2, 5)
    ax5.plot(data['time'], data['vel_x'], 'b-', label='Actual vx', linewidth=2, alpha=0.7)
    ax5.plot(data['time'], data['cmd_vx'], 'b:', label='Command vx', linewidth=2)
    ax5.plot(data['time'], data['vel_y'], 'r-', label='Actual vy', linewidth=2, alpha=0.7)
    ax5.plot(data['time'], data['cmd_vy'], 'r:', label='Command vy', linewidth=2)
    ax5.plot(data['time'], data['vel_z'], 'g-', label='Actual vz', linewidth=2, alpha=0.7)
    ax5.plot(data['time'], data['cmd_vz'], 'g:', label='Command vz', linewidth=2)
    ax5.set_ylabel('Velocity (m/s)', fontsize=10)
    ax5.legend(loc='best', fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Commands vs Actual Velocity', fontsize=11, fontweight='bold')
    
    # ========== PLOT 6: Yaw Tracking ==========
    ax6 = plt.subplot(4, 2, 6)
    ax6.plot(data['time'], np.rad2deg(data['yaw']), 'b-', label='Actual Yaw', linewidth=2)
    ax6.plot(data['time'], np.rad2deg(data['ref_yaw']), 'r--', label='Reference Yaw', linewidth=1.5, alpha=0.7)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(data['time'], np.rad2deg(data['err_yaw']), 'k:', label='Error', linewidth=1.5, alpha=0.5)
    ax6.set_ylabel('Yaw (degrees)', fontsize=10, color='b')
    ax6_twin.set_ylabel('Yaw Error (degrees)', fontsize=10, color='k')
    ax6.legend(loc='upper left', fontsize=8)
    ax6_twin.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Yaw Tracking', fontsize=11, fontweight='bold')
    
    # ========== PLOT 7: MPPI Cost Evolution ==========
    ax7 = plt.subplot(4, 2, 7)
    ax7.plot(data['time'], data['cost_min'], 'g-', label='Min Cost', linewidth=2)
    ax7.plot(data['time'], data['cost_mean'], 'b-', label='Mean Cost', linewidth=2, alpha=0.7)
    
    # Calculate std dev bounds
    cost_upper = data['cost_mean'] + data['cost_std']
    cost_lower = data['cost_mean'] - data['cost_std']
    ax7.fill_between(data['time'], cost_lower, cost_upper, alpha=0.3, label='±1 Std Dev')
    
    ax7.set_ylabel('Cost', fontsize=10)
    ax7.set_xlabel('Time (s)', fontsize=10)
    ax7.legend(loc='best', fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_title('MPPI Cost Evolution', fontsize=11, fontweight='bold')
    ax7.set_yscale('log')  # Log scale for cost
    
    # ========== PLOT 8: ESS and Control Smoothness ==========
    ax8 = plt.subplot(4, 2, 8)
    ax8.plot(data['time'], data['ESS'], 'purple', label='ESS', linewidth=2)
    ax8.set_ylabel('Effective Sample Size', fontsize=10, color='purple')
    ax8.tick_params(axis='y', labelcolor='purple')
    ax8.grid(True, alpha=0.3)
    
    ax8_twin = ax8.twinx()
    ax8_twin.plot(data['time'], data['du'], 'orange', label='Control Change (du)', linewidth=2)
    ax8_twin.set_ylabel('Control Change Magnitude', fontsize=10, color='orange')
    ax8_twin.tick_params(axis='y', labelcolor='orange')
    
    ax8.set_xlabel('Time (s)', fontsize=10)
    ax8.set_title(f'MPPI Statistics (Mean ESS: {np.mean(data["ESS"]):.1f})', 
                  fontsize=11, fontweight='bold')
    ax8.legend(loc='upper left', fontsize=8)
    ax8_twin.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.replace('.csv', '_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # ========== Print Statistics ==========
    print("\n" + "="*60)
    print("FLIGHT STATISTICS")
    print("="*60)
    print(f"Duration:              {data['time'][-1]:.2f} seconds")
    print(f"\nPosition Tracking:")
    print(f"  Mean error:          {np.mean(data['err_pos']):.3f} m")
    print(f"  Max error:           {np.max(data['err_pos']):.3f} m")
    print(f"  Std dev:             {np.std(data['err_pos']):.3f} m")
    print(f"\nVelocity Tracking:")
    print(f"  Mean error:          {np.mean(data['err_vel']):.3f} m/s")
    print(f"  Max error:           {np.max(data['err_vel']):.3f} m/s")
    print(f"  Std dev:             {np.std(data['err_vel']):.3f} m/s")
    print(f"\nYaw Tracking:")
    print(f"  Mean error:          {np.rad2deg(np.mean(np.abs(data['err_yaw']))):.1f} degrees")
    print(f"  Max error:           {np.rad2deg(np.max(np.abs(data['err_yaw']))):.1f} degrees")
    print(f"\nMPPI Performance:")
    print(f"  Mean cost:           {np.mean(data['cost_mean']):.1f}")
    print(f"  Mean ESS:            {np.mean(data['ESS']):.1f}")
    print(f"  Mean control change: {np.mean(data['du']):.3f}")
    print(f"\nControl Saturation:")
    sat_vx = (np.sum(data['sat_vx']) / len(data['sat_vx'])) * 100
    sat_vy = (np.sum(data['sat_vy']) / len(data['sat_vy'])) * 100
    sat_vz = (np.sum(data['sat_vz']) / len(data['sat_vz'])) * 100
    sat_yaw = (np.sum(data['sat_yaw']) / len(data['sat_yaw'])) * 100
    print(f"  Vx saturated:        {sat_vx:.1f}% of time")
    print(f"  Vy saturated:        {sat_vy:.1f}% of time")
    print(f"  Vz saturated:        {sat_vz:.1f}% of time")
    print(f"  Yaw rate saturated:  {sat_yaw:.1f}% of time")
    print("="*60)
    
    plt.show()

def plot_3d_trajectory(csv_path):
    """Plot 3D trajectory"""
    data = load_csv_data(csv_path)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot actual trajectory
    ax.plot(data['pos_x'], data['pos_y'], data['pos_z'], 
            'b-', linewidth=3, label='Actual Trajectory', alpha=0.8)
    
    # Plot reference trajectory
    ax.plot(data['ref_px'], data['ref_py'], data['ref_pz'], 
            'r--', linewidth=2, label='Reference Trajectory', alpha=0.6)
    
    # Mark start and end
    ax.scatter(data['pos_x'][0], data['pos_y'][0], data['pos_z'][0], 
               c='green', s=200, marker='o', label='Start', edgecolors='k', linewidths=2)
    ax.scatter(data['pos_x'][-1], data['pos_y'][-1], data['pos_z'][-1], 
               c='red', s=200, marker='X', label='End', edgecolors='k', linewidths=2)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('3D Flight Trajectory', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    max_range = np.array([
        np.max(data['pos_x']) - np.min(data['pos_x']),
        np.max(data['pos_y']) - np.min(data['pos_y']),
        np.max(data['pos_z']) - np.min(data['pos_z'])
    ]).max() / 2.0
    
    mid_x = (np.max(data['pos_x']) + np.min(data['pos_x'])) * 0.5
    mid_y = (np.max(data['pos_y']) + np.min(data['pos_y'])) * 0.5
    mid_z = (np.max(data['pos_z']) + np.min(data['pos_z'])) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    output_path = csv_path.replace('.csv', '_3d_trajectory.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D trajectory saved to: {output_path}")
    plt.show()

def plot_xy_trajectory(csv_path):
    """Plot 2D top-down view of trajectory"""
    data = load_csv_data(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot actual trajectory
    ax.plot(data['pos_x'], data['pos_y'], 'b-', linewidth=3, label='Actual', alpha=0.8)
    
    # Plot reference trajectory
    ax.plot(data['ref_px'], data['ref_py'], 'r--', linewidth=2, label='Reference', alpha=0.6)
    
    # Mark start and end
    ax.scatter(data['pos_x'][0], data['pos_y'][0], c='green', s=200, 
               marker='o', label='Start', edgecolors='k', linewidths=2, zorder=5)
    ax.scatter(data['pos_x'][-1], data['pos_y'][-1], c='red', s=200, 
               marker='X', label='End', edgecolors='k', linewidths=2, zorder=5)
    
    # Add arrows to show direction every N points
    N = max(1, len(data['pos_x']) // 20)  # ~20 arrows
    for i in range(0, len(data['pos_x'])-1, N):
        dx = data['pos_x'][i+1] - data['pos_x'][i]
        dy = data['pos_y'][i+1] - data['pos_y'][i]
        ax.arrow(data['pos_x'][i], data['pos_y'][i], dx, dy, 
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top-Down View (XY Plane)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    output_path = csv_path.replace('.csv', '_xy_trajectory.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"XY trajectory saved to: {output_path}")
    plt.show()

def main():
    if len(sys.argv) < 2:
        # Try to find most recent log file
        log_dir = os.path.expanduser('/tmp/mppi_logs')
        
        # Search in logs directory first
        if os.path.exists(log_dir):
            log_files = glob.glob(os.path.join(log_dir, 'mppi_log_*.csv'))
        else:
            log_files = []
        
        # Fallback to current directory
        if not log_files:
            log_files = glob.glob('mppi_log_*.csv')
        
        if not log_files:
            print("Usage: python3 plot_mppi_results.py <mppi_log_file.csv>")
            print("No log files found in current directory or ~/mppi_logs/")
            sys.exit(1)
        
        # Use most recent file
        csv_path = max(log_files, key=os.path.getctime)
        print(f"Using most recent log file: {csv_path}")
    else:
        csv_path = sys.argv[1]
    
    # Generate all plots
    plot_mppi_results(csv_path)
    plot_3d_trajectory(csv_path)
    plot_xy_trajectory(csv_path)

if __name__ == '__main__':
    main()