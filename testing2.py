import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CurvingCBF_Robot:
    """Robot that curves around a single obstacle using CBF with angular velocity"""

    def __init__(self, x=0.0, y=0.0, theta=0.0, v_max=0.8, omega_max=3.0):
        self.state = np.array([float(x), float(y), float(theta)], dtype=float)
        self.radius = 0.3
        self.v_max = v_max
        self.omega_max = omega_max

        # SINGLE obstacle placed to force curving around
        self.obstacle = {
            'pos': np.array([4.0, 0.0]),  # Right in the middle of the path
            'radius': 1.0
        }
        self.safety_margin = 0.3  # Reduced for less conservative behavior

        # CBF parameters
        self.gamma = 1.5  # Tuned for smooth curving

        # Goal is BEHIND the obstacle (forces curving around)
        self.goal = np.array([8.0, 2.0])  # Goal is up and to the right

    def dynamics(self, state, u):
        """Differential drive dynamics"""
        x, y, theta = state
        v, omega = u

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        return np.array([dx, dy, dtheta], dtype=float)

    def desired_controller(self):
        """Go toward goal - would go straight through obstacle without CBF"""
        x, y, theta = self.state

        # Vector to goal
        dx_goal = self.goal[0] - x
        dy_goal = self.goal[1] - y
        dist_to_goal = np.sqrt(dx_goal ** 2 + dy_goal ** 2)

        if dist_to_goal < 0.1:
            return np.array([0.0, 0.0], dtype=float)

        # Desired heading toward goal
        desired_theta = np.arctan2(dy_goal, dx_goal)

        # Heading error
        theta_error = desired_theta - theta
        # Normalize to [-pi, pi]
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

        # P-controller for goal tracking
        v_des = min(self.v_max, dist_to_goal * 0.4)
        omega_des = 2.5 * theta_error

        # Limit angular velocity
        omega_des = np.clip(omega_des, -self.omega_max, self.omega_max)

        return np.array([v_des, omega_des], dtype=float)

    def barrier_function(self, state):
        """Barrier function for the single obstacle"""
        x, y, _ = state
        ox, oy = self.obstacle['pos']

        # Euclidean distance to obstacle center
        dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)

        # Safety distance
        safety_distance = self.radius + self.obstacle['radius'] + self.safety_margin

        return dist - safety_distance

    def barrier_gradient_curvature(self, state):
        """
        CRITICAL: Creates a gradient that encourages CURVING around obstacles
        This makes dh/dtheta ≠ 0 in a smart way
        """
        x, y, theta = state
        ox, oy = self.obstacle['pos']

        # Vector from obstacle to robot
        dx = x - ox
        dy = y - oy
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=float)

        # Normal gradient (position component)
        dh_dx = dx / dist
        dh_dy = dy / dist

        # KEY INSIGHT: Create a gradient that encourages turning AWAY
        # based on relative position and heading

        # 1. Compute angle from obstacle to robot
        angle_to_robot = np.arctan2(dy, dx)

        # 2. Compute robot's current heading
        robot_heading = theta

        # 3. Relative angle difference
        rel_angle = robot_heading - angle_to_robot
        # Normalize to [-pi, pi]
        rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))

        # 4. Gradient in theta: Should be POSITIVE when robot points toward obstacle
        # This encourages turning AWAY
        # The closer we are, the stronger the gradient
        proximity_factor = 1.0 / (dist + 0.5)

        # Create a "steering gradient":
        # If robot points toward obstacle (rel_angle ≈ 0), dh_dtheta > 0 encourages right turn
        # If robot points away (rel_angle ≈ ±π), dh_dtheta ≈ 0 (no steering needed)
        dh_dtheta = proximity_factor * np.sin(rel_angle)

        return np.array([dh_dx, dh_dy, dh_dtheta], dtype=float)

    def cbf_safety_filter_with_curving(self, u_des):
        """
        CBF-QP that finds optimal (v, ω) to curve around obstacle
        Solves: min ||u - u_des||² subject to L_g_h·u ≥ -γh
        """
        x = self.state
        u = u_des.copy()

        # Get barrier value and gradient
        h = self.barrier_function(x)
        dh_dx = self.barrier_gradient_curvature(x)

        # If far from obstacle, use desired control
        if h > 1.0:
            return u_des

        # Dynamics components
        cos_theta = np.cos(x[2])
        sin_theta = np.sin(x[2])

        # L_g_h = dh/dx * g(x) where g(x) = [[cosθ, 0], [sinθ, 0], [0, 1]]
        L_g_h = np.array([
            dh_dx[0] * cos_theta + dh_dx[1] * sin_theta,  # For v
            dh_dx[2]  # For ω - THIS IS NOW NON-ZERO!
        ], dtype=float)

        # Required constraint value
        required = -self.gamma * h

        # Current constraint value with desired control
        current_val = np.dot(L_g_h, u_des)

        # DEBUG: Print what's happening
        # print(f"h={h:.3f}, L_g_h=[{L_g_h[0]:.3f}, {L_g_h[1]:.3f}], required={required:.3f}, current={current_val:.3f}")

        if current_val >= required:
            # Desired control already safe
            return u_des

        # Need to modify control - solve 2D QP
        # We want to find u that minimizes ||u - u_des||² subject to L_g_h·u ≥ required

        # This is a convex problem - solve using Lagrange multipliers
        # The solution is: u* = u_des + λ * L_g_h, where λ = max(0, (required - L_g_h·u_des) / ||L_g_h||²)

        L_norm_sq = np.dot(L_g_h, L_g_h)
        if L_norm_sq < 1e-6:
            # No gradient direction - just slow down
            u_safe = np.array([0.0, 0.0], dtype=float)
        else:
            lambda_val = max(0, (required - current_val) / L_norm_sq)
            u_safe = u_des + lambda_val * L_g_h

            # Clip to feasible limits
            u_safe[0] = np.clip(u_safe[0], 0, self.v_max)
            u_safe[1] = np.clip(u_safe[1], -self.omega_max, self.omega_max)

        return u_safe

    def step(self, dt=0.1):
        """Take one simulation step"""
        # Get desired control (go toward goal)
        u_des = self.desired_controller()

        # Apply CBF safety filter that enables curving
        u_safe = self.cbf_safety_filter_with_curving(u_des)

        # Apply dynamics
        derivative = self.dynamics(self.state, u_safe)
        self.state = self.state + derivative * dt

        # Calculate barrier value
        h = self.barrier_function(self.state)

        return u_des, u_safe, h


def run_curving_simulation():
    """Run simulation showing curving around a single obstacle"""

    # Create robot starting with slight upward heading to encourage curving
    robot = CurvingCBF_Robot(x=0.0, y=0.0, theta=0.2, v_max=0.8, omega_max=3.0)

    # Simulation parameters
    dt = 0.1
    max_steps = 400

    # Storage
    states = []
    u_des_history = []
    u_safe_history = []
    barrier_history = []

    print("=" * 70)
    print("CURVING AROUND A SINGLE OBSTACLE WITH CBF")
    print("=" * 70)
    print("\nScenario:")
    print(f"- Single obstacle at ({robot.obstacle['pos'][0]}, {robot.obstacle['pos'][1]})")
    print(f"- Goal at ({robot.goal[0]}, {robot.goal[1]}) - BEHIND the obstacle")
    print(f"- Robot must CURVE AROUND using both v and ω")
    print("\nKey innovation: dh/dtheta ≠ 0 creates steering behavior!")
    print("-" * 70)

    for step in range(max_steps):
        states.append(robot.state.copy())

        u_des, u_safe, h = robot.step(dt)

        u_des_history.append(u_des)
        u_safe_history.append(u_safe)
        barrier_history.append(h)

        if step % 25 == 0:
            print(f"Step {step:3d}: Pos=({robot.state[0]:.2f}, {robot.state[1]:.2f}), "
                  f"θ={robot.state[2]:.2f} rad, "
                  f"v={u_safe[0]:.2f}, ω={u_safe[1]:.2f}, "
                  f"h={h:.3f}")

        # Stop if reached goal
        if np.linalg.norm(robot.state[:2] - robot.goal) < 0.5:
            print(f"\n🎯 Reached goal at step {step}!")
            break

        if h < -0.3:
            print(f"\n⚠️  Collision at step {step}! h = {h:.3f}")
            break

    print(f"\nSimulation ended after {len(states)} steps")

    # Convert to arrays
    states = np.array(states)
    u_des = np.array(u_des_history)
    u_safe = np.array(u_safe_history)
    barrier = np.array(barrier_history)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))

    # 1. Main trajectory plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax1.plot(states[0, 0], states[0, 1], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax1.plot(states[-1, 0], states[-1, 1], 'ro', markersize=12, label='End', markeredgecolor='black')
    ax1.plot(robot.goal[0], robot.goal[1], 'm*', markersize=20, label='Goal')

    # Draw obstacle
    obstacle_circle = plt.Circle(robot.obstacle['pos'], robot.obstacle['radius'],
                                 color='red', alpha=0.5, label='Obstacle')
    ax1.add_patch(obstacle_circle)

    # Safety boundary
    safety_circle = plt.Circle(robot.obstacle['pos'],
                               robot.obstacle['radius'] + robot.radius + robot.safety_margin,
                               color='orange', alpha=0.2, linestyle='--',
                               fill=False, label='Safety boundary')
    ax1.add_patch(safety_circle)

    # Draw robot at several points to show orientation
    for i in range(0, len(states), len(states) // 10):
        if i < len(states):
            x, y, theta = states[i]
            # Robot circle
            robot_point = plt.Circle((x, y), robot.radius / 3, color='blue', alpha=0.3)
            ax1.add_patch(robot_point)
            # Heading arrow
            arrow_len = robot.radius
            ax1.arrow(x, y, arrow_len * np.cos(theta), arrow_len * np.sin(theta),
                      head_width=0.1, head_length=0.15, fc='darkblue', ec='darkblue', alpha=0.5)

    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Curving Around Obstacle: Trajectory')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_xlim(-1, 9)
    ax1.set_ylim(-2, 4)

    # 2. Control inputs over time
    ax2 = plt.subplot(2, 3, 2)
    time = np.arange(len(u_des)) * dt
    ax2.plot(time, u_des[:, 0], 'r--', alpha=0.7, linewidth=1.5, label='Desired v')
    ax2.plot(time, u_safe[:, 0], 'b-', linewidth=2, label='Safe v')
    ax2.plot(time, u_safe[:, 1], 'g-', linewidth=2, label='Safe ω')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control inputs')
    ax2.set_title('Control Inputs: v and ω Working Together')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Heading angle evolution
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time, states[:len(time), 2], 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heading θ (rad)')
    ax3.set_title('Robot Heading Evolution')
    ax3.grid(True, alpha=0.3)

    # 4. Barrier function
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time, barrier, 'g-', linewidth=2, label='h(x)')
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label='Safety boundary')

    # Fill safe/unsafe regions
    safe_mask = barrier >= 0
    unsafe_mask = barrier < 0

    if np.any(safe_mask):
        ax4.fill_between(time[safe_mask], 0, barrier[safe_mask],
                         alpha=0.3, color='green', label='Safe region')
    if np.any(unsafe_mask):
        ax4.fill_between(time[unsafe_mask], barrier[unsafe_mask], 0,
                         alpha=0.3, color='red', label='Unsafe violation')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Barrier function h(x)')
    ax4.set_title('Safety Margin: Curving Keeps h(x) ≥ 0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Phase plot: ω vs v
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(u_safe[:, 0], u_safe[:, 1], c=time, cmap='viridis',
                          alpha=0.6, s=30)
    ax5.set_xlabel('Linear velocity (v)')
    ax5.set_ylabel('Angular velocity (ω)')
    ax5.set_title('Control Phase Space: v-ω Relationship')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Time (s)')

    # 6. Distance to obstacle over time
    ax6 = plt.subplot(2, 3, 6)
    distances = []
    for state in states:
        dist = np.linalg.norm(state[:2] - robot.obstacle['pos'])
        distances.append(dist)
    distances = np.array(distances)

    ax6.plot(time, distances, 'b-', linewidth=2, label='Distance to obstacle')
    ax6.axhline(y=robot.obstacle['radius'] + robot.radius,
                color='r', linestyle='--', alpha=0.7, label='Collision distance')
    ax6.axhline(y=robot.obstacle['radius'] + robot.radius + robot.safety_margin,
                color='orange', linestyle='--', alpha=0.7, label='Safety boundary')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Distance')
    ax6.set_title('Distance to Obstacle Center')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create animation
    print("\n" + "=" * 70)
    print("CREATING ANIMATION...")
    print("=" * 70)

    # Remove ALL animation code and replace with this simple static plot:

    print("\n" + "=" * 70)
    print("CREATING TRAJECTORY PLOT...")
    print("=" * 70)

    # Create a simple static trajectory plot
    fig_traj, ax_traj = plt.subplots(figsize=(10, 8))

    # Plot the entire trajectory
    ax_traj.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax_traj.plot(states[0, 0], states[0, 1], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax_traj.plot(states[-1, 0], states[-1, 1], 'ro', markersize=12, label='End', markeredgecolor='black')
    ax_traj.plot(robot.goal[0], robot.goal[1], 'm*', markersize=20, label='Goal')

    # Draw obstacle
    obstacle_circle = plt.Circle(robot.obstacle['pos'], robot.obstacle['radius'],
                                 color='red', alpha=0.5, label='Obstacle')
    ax_traj.add_patch(obstacle_circle)

    # Safety boundary
    safety_circle = plt.Circle(robot.obstacle['pos'],
                               robot.obstacle['radius'] + robot.radius + robot.safety_margin,
                               color='orange', alpha=0.2, linestyle='--',
                               fill=False, label='Safety boundary')
    ax_traj.add_patch(safety_circle)

    # Draw robot at several points to show orientation (optional)
    for i in range(0, len(states), len(states) // 15):  # Fewer points for clarity
        if i < len(states):
            x, y, theta = states[i]
            # Small dot for position
            ax_traj.plot(x, y, 'b.', markersize=4, alpha=0.5)
            # Heading arrow
            arrow_len = robot.radius
            ax_traj.arrow(x, y, arrow_len * np.cos(theta), arrow_len * np.sin(theta),
                          head_width=0.08, head_length=0.1, fc='darkblue', ec='darkblue', alpha=0.4)

    # Formatting
    ax_traj.set_xlabel('X position')
    ax_traj.set_ylabel('Y position')
    ax_traj.set_title('CBF-Controlled Robot: Curving Around Single Obstacle')
    ax_traj.axis('equal')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc='upper left')
    ax_traj.set_xlim(-1, 9)
    ax_traj.set_ylim(-2, 4)

    # Add info box
    info_text = (
        f"Total steps: {len(states)}\n"
        f"Start: ({states[0, 0]:.1f}, {states[0, 1]:.1f})\n"
        f"End: ({states[-1, 0]:.1f}, {states[-1, 1]:.1f})\n"
        f"Min safety h(x): {barrier.min():.3f}\n"
        f"Final v: {u_safe[-1, 0]:.2f}, ω: {u_safe[-1, 1]:.2f}"
    )
    ax_traj.text(0.02, 0.02, info_text, transform=ax_traj.transAxes,
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    print("\nDisplaying trajectory plot...")
    plt.show()


# Run the simulation
if __name__ == "__main__":
    run_curving_simulation()