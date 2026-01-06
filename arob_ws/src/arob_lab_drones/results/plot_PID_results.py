#!/usr/bin/env python3
"""
Plot PID flight results from CSV log
Usage: python3 plot_PID_results.py pid_log_1234567890.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import csv
import glob

def load_csv_data(csv_path):
    """Load CSV data into dictionary of lists"""
    data = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for fieldname in reader.fieldnames:
            data[fieldname] = []
        
        for row in reader:
            for key, value in row.items():
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)
    
    for key in data:
        if key != 'control_mode':
            data[key] = np.array(data[key])
    
    return data

def plot_pid_results(csv_path):
    """Generate comprehensive plots from PID log CSV"""
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    data = load_csv_data(csv_path)
    print(f"Loaded {len(data['time'])} data points from {csv_path}")
    print(f"Flight duration: {data['time'][-1]:.2f} seconds")
    print(f"Control mode: {data['control_mode'][0]}")
    
    is_velocity_mode = data['control_mode'][0] == 'velocity'
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ========== PLOT 1: Position Tracking ==========
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(data['time'], data['pos_x'], 'b-', label='Actual X', linewidth=2)
    ax1.plot(data['time'], data['ref_px'], 'b--', label='Reference X', linewidth=1.5, alpha=0.7)
    ax1.plot(data['time'], data['pos_y'], 'r-', label='Actual Y', linewidth=2)
    ax1.plot(data['time'], data['ref_py'], 'r--', label='Reference Y', linewidth=1.5, alpha=0.7)
    ax1.plot(data['time'], data['pos_z'], 'g-', label='Actual Z', linewidth=2)
    ax1.plot(data['time'], data['ref_pz'], 'g--', label='Reference Z', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Position (m)', fontsize=10)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Position Tracking - {data["control_mode"][0].upper()} Mode', fontsize=11, fontweight='bold')
    
    # ========== PLOT 2: Position Error ==========
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(data['time'], data['err_pos'], 'k-', linewidth=2)
    ax2.fill_between(data['time'], 0, data['err_pos'], alpha=0.3)
    ax2.set_ylabel('Position Error (m)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Position Error (Mean: {np.mean(data["err_pos"]):.3f}m, Max: {np.max(data["err_pos"]):.3f}m)', 
                  fontsize=11, fontweight='bold')
    
    # ========== PLOT 3: Velocity Tracking ==========
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(data['time'], data['vel_x'], 'b-', label='Actual vx', linewidth=2)
    ax3.plot(data['time'], data['ref_vx'], 'b--', label='Reference vx', linewidth=1.5, alpha=0.7)
    ax3.plot(data['time'], data['vel_y'], 'r-', label='Actual vy', linewidth=2)
    ax3.plot(data['time'], data['ref_vy'], 'r--', label='Reference vy', linewidth=1.5, alpha=0.7)
    ax3.plot(data['time'], data['vel_z'], 'g-', label='Actual vz', linewidth=2)
    ax3.plot(data['time'], data['ref_vz'], 'g--', label='Reference vz', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Velocity (m/s)', fontsize=10)
    ax3.legend(loc='best', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Velocity Tracking', fontsize=11, fontweight='bold')
    
    # ========== PLOT 4: Velocity Error ==========
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(data['time'], data['err_vel'], 'k-', linewidth=2)
    ax4.fill_between(data['time'], 0, data['err_vel'], alpha=0.3)
    ax4.set_ylabel('Velocity Error (m/s)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f'Velocity Error (Mean: {np.mean(data["err_vel"]):.3f}m/s, Max: {np.max(data["err_vel"]):.3f}m/s)', 
                  fontsize=11, fontweight='bold')
    
    # ========== PLOT 5: Commands ==========
    ax5 = plt.subplot(3, 2, 5)
    if is_velocity_mode:
        ax5.plot(data['time'], data['cmd_vx'], 'b-', label='Cmd vx', linewidth=2)
        ax5.plot(data['time'], data['cmd_vy'], 'r-', label='Cmd vy', linewidth=2)
        ax5.plot(data['time'], data['cmd_vz'], 'g-', label='Cmd vz', linewidth=2)
        ax5.set_ylabel('Velocity Commands (m/s)', fontsize=10)
        ax5.set_title('Velocity Commands (PID Output)', fontsize=11, fontweight='bold')
    else:
        ax5.plot(data['time'], data['cmd_px'], 'b-', label='Cmd px', linewidth=2)
        ax5.plot(data['time'], data['cmd_py'], 'r-', label='Cmd py', linewidth=2)
        ax5.plot(data['time'], data['cmd_pz'], 'g-', label='Cmd pz', linewidth=2)
        ax5.set_ylabel('Position Commands (m)', fontsize=10)
        ax5.set_title('Position Commands (PID Output)', fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('Time (s)', fontsize=10)
    
    # ========== PLOT 6: Integral Terms (Position Mode Only) ==========
    ax6 = plt.subplot(3, 2, 6)
    if not is_velocity_mode and np.any(data['integral_x'] != 0):
        ax6.plot(data['time'], data['integral_x'], 'b-', label='Integral X', linewidth=2)
        ax6.plot(data['time'], data['integral_y'], 'r-', label='Integral Y', linewidth=2)
        ax6.plot(data['time'], data['integral_z'], 'g-', label='Integral Z', linewidth=2)
        ax6.set_ylabel('Integral Term (m)', fontsize=10)
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_title('PID Integral Terms', fontsize=11, fontweight='bold')
    else:
        # Show control gains used
        if is_velocity_mode:
            gains_text = "Velocity Mode Gains:\nkp = 1.0\nkv = 2.0\nka = 0.5"
        else:
            gains_text = "Position Mode Gains:\nkp = 0.75\nkd = 3.0\nki = 0.05"
        ax6.text(0.5, 0.5, gains_text, ha='center', va='center', 
                fontsize=14, transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.axis('off')
        ax6.set_title('Controller Gains', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time (s)', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.replace('.csv', '_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # ========== Print Statistics ==========
    print("\n" + "="*60)
    print("FLIGHT STATISTICS")
    print("="*60)
    print(f"Control Mode:          {data['control_mode'][0].upper()}")
    print(f"Duration:              {data['time'][-1]:.2f} seconds")
    print(f"\nPosition Tracking:")
    print(f"  Mean error:          {np.mean(data['err_pos']):.3f} m")
    print(f"  Max error:           {np.max(data['err_pos']):.3f} m")
    print(f"  Std dev:             {np.std(data['err_pos']):.3f} m")
    print(f"\nVelocity Tracking:")
    print(f"  Mean error:          {np.mean(data['err_vel']):.3f} m/s")
    print(f"  Max error:           {np.max(data['err_vel']):.3f} m/s")
    print(f"  Std dev:             {np.std(data['err_vel']):.3f} m/s")
    print("="*60)
    
    plt.show()

def plot_3d_trajectory(csv_path):
    """Plot 3D trajectory"""
    data = load_csv_data(csv_path)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(data['pos_x'], data['pos_y'], data['pos_z'], 
            'b-', linewidth=3, label='Actual Trajectory', alpha=0.8)
    ax.plot(data['ref_px'], data['ref_py'], data['ref_pz'], 
            'r--', linewidth=2, label='Reference Trajectory', alpha=0.6)
    
    ax.scatter(data['pos_x'][0], data['pos_y'][0], data['pos_z'][0], 
               c='green', s=200, marker='o', label='Start', edgecolors='k', linewidths=2)
    ax.scatter(data['pos_x'][-1], data['pos_y'][-1], data['pos_z'][-1], 
               c='red', s=200, marker='X', label='End', edgecolors='k', linewidths=2)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'3D Flight Trajectory - {data["control_mode"][0].upper()} Mode', 
                 fontsize=14, fontweight='bold')
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

def main():
    if len(sys.argv) < 2:
        log_dir = os.path.expanduser('~/pid_logs')
        
        if os.path.exists(log_dir):
            log_files = glob.glob(os.path.join(log_dir, 'pid_log_*.csv'))
        else:
            log_files = []
        
        if not log_files:
            log_files = glob.glob('pid_log_*.csv')
        
        if not log_files:
            print("Usage: python3 plot_PID_results.py <pid_log_file.csv>")
            print("No log files found in current directory or ~/pid_logs/")
            sys.exit(1)
        
        csv_path = max(log_files, key=os.path.getctime)
        print(f"Using most recent log file: {csv_path}")
    else:
        csv_path = sys.argv[1]
    
    plot_pid_results(csv_path)
    plot_3d_trajectory(csv_path)

if __name__ == '__main__':
    main()