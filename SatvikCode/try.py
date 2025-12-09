#!/usr/bin/env python3
"""
Control Barrier Functions for Multi-Robot Collision Avoidance
EECE5550 Mobile Robotics Project
Authors: Satvik Tajane, Harsh Akabari, Nicolas Drager

Run with: python3 cbf_simulation.py
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


# ============================================================================
# SIMULATION PARAMETERS (HYPERPARAMETERS - TUNE THESE!)
# ============================================================================

class SimParams:
    # TIME PARAMETERS
    dt = 0.05           # Time step (seconds)
    T_max = 30.0        # Maximum simulation time (seconds)
    
    # ROBOT PARAMETERS
    n_robots = 2        # Number of robots
    robot_radius = 0.15 # Robot radius (meters)
    v_max = 0.22        # Maximum velocity (m/s) - TurtleBot3 limit
    
    # CBF PARAMETERS
    gamma = 3.0         # CBF decay rate
    D_safe = 0.4        # Minimum safe distance between robots (meters)
    
    # REFERENCE CONTROLLER PARAMETERS
    Kp = 0.5            # Proportional gain for goal tracking
    goal_tolerance = 0.1


# ============================================================================
# ROBOT CLASS
# ============================================================================

class Robot:
    def __init__(self, robot_id, initial_pos, goal_pos, color):
        self.id = robot_id
        self.pos = np.array(initial_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.color = color
        self.pos_history = [self.pos.copy()]
        self.vel_history = [self.vel.copy()]
        self.reached_goal = False
    
    def update(self, velocity, dt):
        self.vel = velocity
        self.pos = self.pos + velocity * dt
        self.pos_history.append(self.pos.copy())
        self.vel_history.append(self.vel.copy())
        if np.linalg.norm(self.pos - self.goal) < SimParams.goal_tolerance:
            self.reached_goal = True
    
    def get_desired_velocity(self):
        if self.reached_goal:
            return np.array([0.0, 0.0])
        to_goal = self.goal - self.pos
        distance = np.linalg.norm(to_goal)
        if distance < 0.01:
            return np.array([0.0, 0.0])
        v_desired = SimParams.Kp * to_goal
        speed = np.linalg.norm(v_desired)
        if speed > SimParams.v_max:
            v_desired = v_desired / speed * SimParams.v_max
        return v_desired


# ============================================================================
# CBF FUNCTIONS
# ============================================================================

def compute_barrier_function(robot_i, robot_j):
    p_rel = robot_i.pos - robot_j.pos
    distance_sq = np.dot(p_rel, p_rel)
    h = distance_sq - SimParams.D_safe ** 2
    return h


def cbf_safety_filter(robot_i, all_robots, u_desired):
    u = cp.Variable(2)
    constraints = []
    barrier_values = {}
    
    for robot_j in all_robots:
        if robot_j.id == robot_i.id:
            continue
        p_rel = robot_i.pos - robot_j.pos
        h = compute_barrier_function(robot_i, robot_j)
        barrier_values[f"h_{robot_i.id}{robot_j.id}"] = h
        lhs = 2 * p_rel @ u
        rhs = -SimParams.gamma * h + 2 * np.dot(p_rel, robot_j.vel)
        constraints.append(lhs >= rhs)
    
    constraints.append(u[0] >= -SimParams.v_max)
    constraints.append(u[0] <= SimParams.v_max)
    constraints.append(u[1] >= -SimParams.v_max)
    constraints.append(u[1] <= SimParams.v_max)
    
    objective = cp.Minimize(cp.sum_squares(u - u_desired))
    problem = cp.Problem(objective, constraints)
    
    solvers_to_try = [cp.OSQP, cp.ECOS, cp.SCS]
    u_safe = None
    
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                u_safe = u.value
                break
        except Exception:
            continue
    
    if u_safe is None:
        print(f"Warning: QP failed for robot {robot_i.id}, using zero velocity")
        u_safe = np.array([0.0, 0.0])
    
    return u_safe, barrier_values


# ============================================================================
# SIMULATION LOOP
# ============================================================================

def run_simulation(robots, use_cbf=True):
    time_history = [0.0]
    barrier_history = {f"h_{i}{j}": [] for i in range(len(robots)) 
                       for j in range(len(robots)) if i != j}
    min_distance_history = []
    
    t = 0.0
    step = 0
    
    print(f"\n{'='*60}")
    print(f"Running simulation {'WITH' if use_cbf else 'WITHOUT'} CBF")
    print(f"{'='*60}")
    
    while t < SimParams.T_max:
        if all(r.reached_goal for r in robots):
            print(f"All robots reached goals at t = {t:.2f}s")
            break
        
        min_dist = float('inf')
        for i, robot_i in enumerate(robots):
            for j, robot_j in enumerate(robots):
                if i < j:
                    dist = np.linalg.norm(robot_i.pos - robot_j.pos)
                    min_dist = min(min_dist, dist)
        min_distance_history.append(min_dist)
        
        all_barrier_values = {}
        for robot in robots:
            u_desired = robot.get_desired_velocity()
            if use_cbf:
                u_safe, barrier_values = cbf_safety_filter(robot, robots, u_desired)
                all_barrier_values.update(barrier_values)
            else:
                u_safe = u_desired
            robot.update(u_safe, SimParams.dt)
        
        for key in barrier_history:
            if key in all_barrier_values:
                barrier_history[key].append(all_barrier_values[key])
        
        t += SimParams.dt
        time_history.append(t)
        step += 1
        
        if step % 100 == 0:
            print(f"t = {t:.2f}s, min_distance = {min_dist:.3f}m")
    
    return time_history, barrier_history, min_distance_history


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_animation(robots_cbf, robots_no_cbf, time_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, title in [(ax1, 'WITH CBF (Safe)'), (ax2, 'WITHOUT CBF (Collision)')]:
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    for robots, ax in [(robots_cbf, ax1), (robots_no_cbf, ax2)]:
        for robot in robots:
            ax.plot(robot.goal[0], robot.goal[1], 'x', color=robot.color, 
                   markersize=15, markeredgewidth=3, label=f'Goal {robot.id}')
    
    circles_cbf, circles_no_cbf = [], []
    trails_cbf, trails_no_cbf = [], []
    
    for robot in robots_cbf:
        circle = Circle(robot.pos_history[0], SimParams.robot_radius, 
                       color=robot.color, alpha=0.7)
        ax1.add_patch(circle)
        circles_cbf.append(circle)
        trail, = ax1.plot([], [], '-', color=robot.color, alpha=0.5, linewidth=2)
        trails_cbf.append(trail)
    
    for robot in robots_no_cbf:
        circle = Circle(robot.pos_history[0], SimParams.robot_radius, 
                       color=robot.color, alpha=0.7)
        ax2.add_patch(circle)
        circles_no_cbf.append(circle)
        trail, = ax2.plot([], [], '-', color=robot.color, alpha=0.5, linewidth=2)
        trails_no_cbf.append(trail)
    
    time_text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    time_text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.add_patch(Circle((1.5, 1.5), SimParams.D_safe/2, fill=False, 
                         linestyle='--', color='gray', alpha=0.5))
    ax1.text(1.5, 1.5 + SimParams.D_safe/2 + 0.1, f'D_safe={SimParams.D_safe}m', 
             ha='center', fontsize=9, color='gray')
    
    def animate(frame):
        for i, robot in enumerate(robots_cbf):
            pos_idx = min(frame, len(robot.pos_history) - 1)
            circles_cbf[i].center = robot.pos_history[pos_idx]
            trail_x = [p[0] for p in robot.pos_history[:pos_idx+1]]
            trail_y = [p[1] for p in robot.pos_history[:pos_idx+1]]
            trails_cbf[i].set_data(trail_x, trail_y)
        
        for i, robot in enumerate(robots_no_cbf):
            pos_idx = min(frame, len(robot.pos_history) - 1)
            circles_no_cbf[i].center = robot.pos_history[pos_idx]
            trail_x = [p[0] for p in robot.pos_history[:pos_idx+1]]
            trail_y = [p[1] for p in robot.pos_history[:pos_idx+1]]
            trails_no_cbf[i].set_data(trail_x, trail_y)
        
        t = frame * SimParams.dt
        time_text1.set_text(f't = {t:.2f}s\nγ = {SimParams.gamma}')
        time_text2.set_text(f't = {t:.2f}s\nNo CBF')
        
        return circles_cbf + circles_no_cbf + trails_cbf + trails_no_cbf + [time_text1, time_text2]
    
    n_frames = max(len(robots_cbf[0].pos_history), len(robots_no_cbf[0].pos_history))
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)
    plt.tight_layout()
    return fig, anim


def plot_analysis(robots, time_history, min_distance_history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for robot in robots:
        traj = np.array(robot.pos_history)
        ax.plot(traj[:, 0], traj[:, 1], '-', color=robot.color, 
               linewidth=2, label=f'Robot {robot.id}')
        ax.plot(robot.pos_history[0][0], robot.pos_history[0][1], 'o', 
               color=robot.color, markersize=10)
        ax.plot(robot.goal[0], robot.goal[1], 'x', color=robot.color, 
               markersize=15, markeredgewidth=3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Robot Trajectories', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    ax = axes[0, 1]
    t = np.array(time_history[:len(min_distance_history)])
    ax.plot(t, min_distance_history, 'b-', linewidth=2)
    ax.axhline(y=SimParams.D_safe, color='r', linestyle='--', 
               label=f'D_safe = {SimParams.D_safe}m')
    ax.axhline(y=2*SimParams.robot_radius, color='orange', linestyle=':', 
               label=f'Collision = {2*SimParams.robot_radius}m')
    ax.fill_between(t, 0, SimParams.D_safe, alpha=0.2, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Minimum Distance (m)')
    ax.set_title('Minimum Inter-Robot Distance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for robot in robots:
        vel_mag = [np.linalg.norm(v) for v in robot.vel_history]
        t = np.arange(len(vel_mag)) * SimParams.dt
        ax.plot(t, vel_mag, '-', color=robot.color, linewidth=2, 
               label=f'Robot {robot.id}')
    ax.axhline(y=SimParams.v_max, color='gray', linestyle='--', 
               label=f'v_max = {SimParams.v_max}m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Robot Velocities', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for robot in robots:
        dist_to_goal = [np.linalg.norm(np.array(p) - robot.goal) 
                        for p in robot.pos_history]
        t = np.arange(len(dist_to_goal)) * SimParams.dt
        ax.plot(t, dist_to_goal, '-', color=robot.color, linewidth=2, 
               label=f'Robot {robot.id}')
    ax.axhline(y=SimParams.goal_tolerance, color='green', linestyle='--',
               label=f'Goal tolerance = {SimParams.goal_tolerance}m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to Goal (m)')
    ax.set_title('Goal Convergence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("="*70)
    print("CONTROL BARRIER FUNCTIONS - TWO-ROBOT COLLISION AVOIDANCE")
    print("="*70)
    print(f"\nSimulation Parameters:")
    print(f"  Number of robots: {SimParams.n_robots}")
    print(f"  CBF gamma: {SimParams.gamma}")
    print(f"  Safe distance: {SimParams.D_safe}m")
    print(f"  Max velocity: {SimParams.v_max}m/s")
    print(f"  Robot radius: {SimParams.robot_radius}m")
    
    # TWO ROBOTS: Crossing paths (swap positions diagonally)
    robot_configs = [
        {'id': 0, 'start': [0.0, 0.0], 'goal': [3.0, 3.0], 'color': 'blue'},
        {'id': 1, 'start': [3.0, 0.0], 'goal': [0.0, 3.0], 'color': 'red'},
    ]
    
    robots_cbf = [Robot(cfg['id'], cfg['start'], cfg['goal'], cfg['color']) 
                  for cfg in robot_configs]
    robots_no_cbf = [Robot(cfg['id'], cfg['start'], cfg['goal'], cfg['color']) 
                    for cfg in robot_configs]
    
    print("\n" + "="*70)
    time_cbf, barrier_cbf, min_dist_cbf = run_simulation(robots_cbf, use_cbf=True)
    time_no_cbf, _, min_dist_no_cbf = run_simulation(robots_no_cbf, use_cbf=False)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    min_dist_cbf_overall = min(min_dist_cbf) if min_dist_cbf else float('inf')
    min_dist_no_cbf_overall = min(min_dist_no_cbf) if min_dist_no_cbf else float('inf')
    
    print(f"\nWITH CBF:")
    print(f"  Minimum distance achieved: {min_dist_cbf_overall:.4f}m")
    print(f"  Safety margin (D_safe): {SimParams.D_safe}m")
    if min_dist_cbf_overall >= SimParams.D_safe - 0.01:
        print(f"  ✓ SAFETY MAINTAINED!")
    else:
        print(f"  ✗ Safety violation by {SimParams.D_safe - min_dist_cbf_overall:.4f}m")
    
    print(f"\nWITHOUT CBF:")
    print(f"  Minimum distance achieved: {min_dist_no_cbf_overall:.4f}m")
    if min_dist_no_cbf_overall < 2 * SimParams.robot_radius:
        print(f"  ✗ COLLISION OCCURRED!")
    else:
        print(f"  No collision (got lucky with timing)")
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS...")
    print("="*70)
    
    fig_anim, anim = create_animation(robots_cbf, robots_no_cbf, time_cbf)
    fig_analysis = plot_analysis(robots_cbf, time_cbf, min_dist_cbf)
    
    plt.show()
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    
    return robots_cbf, time_cbf, barrier_cbf


if __name__ == "__main__":
    robots, time_history, barrier_history = main()