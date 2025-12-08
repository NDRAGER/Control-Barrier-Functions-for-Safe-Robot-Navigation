import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


class DifferentialDriveRobot:
    """Simple differential drive robot with CBF safety controller"""

    def __init__(self, x=0.0, y=0.0, theta=0.0, v_max=0.5, omega_max=2.0):
        # State: [x, y, theta] - explicitly make them floats
        self.state = np.array([float(x), float(y), float(theta)], dtype=float)

        # Robot parameters
        self.radius = 0.3  # Robot radius
        self.v_max = v_max  # Max linear velocity
        self.omega_max = omega_max  # Max angular velocity

        # Obstacle (static for simplicity)
        self.obstacle_pos = np.array([5.0, 0.0], dtype=float)
        self.obstacle_radius = 1.0
        self.safety_margin = 0.5  # Additional safety margin

        # CBF parameters
        self.gamma = 1.0  # CBF tuning parameter

    def dynamics(self, state, u):
        """Differential drive dynamics: [x, y, theta]"""
        x, y, theta = state
        v, omega = u

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        return np.array([dx, dy, dtheta], dtype=float)

    def desired_controller(self):
        """Nominal (unsafe) controller: go straight ahead"""
        # This is intentionally unsafe - just goes forward
        return np.array([self.v_max, 0.0], dtype=float)  # [v, omega]

    def barrier_function(self, state):
        """Barrier function: distance to obstacle minus safety radius"""
        x, y, _ = state
        # Distance from robot center to obstacle center
        dist = np.linalg.norm([x - self.obstacle_pos[0],
                               y - self.obstacle_pos[1]])
        # Subtract robot and obstacle radii plus safety margin
        return dist - (self.radius + self.obstacle_radius + self.safety_margin)

    def barrier_gradient(self, state):
        """Gradient of barrier function"""
        x, y, _ = state
        dist_vec = np.array([x - self.obstacle_pos[0],
                             y - self.obstacle_pos[1]])
        dist = np.linalg.norm(dist_vec)

        if dist < 1e-6:  # Avoid division by zero
            return np.array([1.0, 0.0, 0.0], dtype=float)

        dh_dx = dist_vec[0] / dist
        dh_dy = dist_vec[1] / dist
        dh_dtheta = 0.0  # Barrier doesn't depend on orientation

        return np.array([dh_dx, dh_dy, dh_dtheta], dtype=float)

    def cbf_safety_filter(self, u_des):
        """CBF-QP safety filter"""
        # Get current state
        x = self.state

        # Compute barrier value and gradient
        h = self.barrier_function(x)
        dh_dx = self.barrier_gradient(x)

        # Drift term: L_f h = dh/dx * f(x) with u=0
        # For our system, f(x) = [v*cosθ, v*sinθ, ω], but at u=0, f(x)=0
        # So L_f h = 0 for stationary dynamics
        L_f_h = 0.0

        # Control matrix: L_g h = dh/dx * g(x)
        # g(x) for differential drive:
        # g(x) = [[cosθ, 0], [sinθ, 0], [0, 1]]
        cos_theta = np.cos(x[2])
        sin_theta = np.sin(x[2])
        L_g_h = np.array([
            dh_dx[0] * cos_theta + dh_dx[1] * sin_theta,  # For v
            dh_dx[2]  # For omega (but dh_dtheta = 0, so this is 0)
        ], dtype=float)

        # CBF constraint: L_f_h + L_g_h * u ≥ -gamma * h
        # Since L_f_h = 0 and L_g_h[1] = 0, we have:
        # L_g_h[0] * v ≥ -gamma * h

        # Check if desired control already satisfies CBF constraint
        constraint_val = L_g_h[0] * u_des[0]

        if constraint_val >= -self.gamma * h:
            # Desired control is safe
            return u_des
        else:
            # Need to modify control to satisfy constraint
            # Simple solution: reduce forward velocity

            if abs(L_g_h[0]) > 1e-6:  # Avoid division by zero
                v_safe = max(-self.gamma * h / L_g_h[0], 0)
            else:
                v_safe = 0.0

            # Ensure within bounds
            v_safe = np.clip(v_safe, 0, self.v_max)

            # Return safe control (keep desired angular velocity)
            return np.array([v_safe, u_des[1]], dtype=float)

    def step(self, dt=0.1):
        """Take one simulation step"""
        # Get desired (unsafe) control
        u_des = self.desired_controller()

        # Apply CBF safety filter
        u_safe = self.cbf_safety_filter(u_des)

        # Apply dynamics
        derivative = self.dynamics(self.state, u_safe)
        self.state = self.state + derivative * dt

        # Ensure state stays as float
        self.state = self.state.astype(float)

        return u_des, u_safe, self.barrier_function(self.state)


class CBFSimulator:
    """Visual simulator for CBF-controlled robot"""

    def __init__(self):
        # Create robot starting at (0, 0) facing east
        self.robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)

        # Simulation parameters
        self.dt = 0.1
        self.max_steps = 200

        # Data recording
        self.history = {
            'states': [],
            'u_des': [],
            'u_safe': [],
            'barrier': []
        }

    def run_simulation(self):
        """Run the simulation"""
        print("Starting CBF simulation...")
        print(f"Robot starting at: ({self.robot.state[0]:.1f}, {self.robot.state[1]:.1f})")
        print(
            f"Obstacle at: ({self.robot.obstacle_pos[0]:.1f}, {self.robot.obstacle_pos[1]:.1f}), radius: {self.robot.obstacle_radius}")
        print(f"Safety margin: {self.robot.safety_margin}")
        print("-" * 50)

        for step in range(self.max_steps):
            # Record current state
            self.history['states'].append(self.robot.state.copy())

            # Take step
            u_des, u_safe, h = self.robot.step(self.dt)

            # Record data
            self.history['u_des'].append(u_des)
            self.history['u_safe'].append(u_safe)
            self.history['barrier'].append(h)

            # Print info every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: Pos=({self.robot.state[0]:.2f}, {self.robot.state[1]:.2f}), "
                      f"h={h:.3f}, "
                      f"v_des={u_des[0]:.2f}, v_safe={u_safe[0]:.2f}")

            # Stop if barrier function becomes negative (collision)
            if h < -0.1:
                print(f"\n⚠️  Safety violation at step {step}! h = {h:.3f}")
                break

        print("\nSimulation complete!")
        return len(self.history['states'])

    def plot_results(self):
        """Plot the simulation results"""
        states = np.array(self.history['states'])
        u_des = np.array(self.history['u_des'])
        u_safe = np.array(self.history['u_safe'])
        barrier = np.array(self.history['barrier'])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Trajectory
        ax1 = axes[0, 0]
        ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax1.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')

        # Draw robot at final position
        final_robot = plt.Circle((states[-1, 0], states[-1, 1]),
                                 self.robot.radius,
                                 color='blue', alpha=0.3)
        ax1.add_patch(final_robot)

        # Draw obstacle
        obstacle = plt.Circle((self.robot.obstacle_pos[0],
                               self.robot.obstacle_pos[1]),
                              self.robot.obstacle_radius,
                              color='red', alpha=0.5, label='Obstacle')
        ax1.add_patch(obstacle)

        # Draw safety boundary
        safety_boundary = plt.Circle((self.robot.obstacle_pos[0],
                                      self.robot.obstacle_pos[1]),
                                     self.robot.obstacle_radius +
                                     self.robot.radius +
                                     self.robot.safety_margin,
                                     color='orange', alpha=0.2,
                                     linestyle='--', fill=False,
                                     label='Safety boundary')
        ax1.add_patch(safety_boundary)

        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_title('Robot Trajectory with CBF Safety')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Barrier function over time
        ax2 = axes[0, 1]
        steps = np.arange(len(barrier))
        ax2.plot(steps, barrier, 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Safety boundary (h=0)')

        # Safe/unsafe regions
        safe_mask = barrier >= 0
        unsafe_mask = barrier < 0

        if np.any(safe_mask):
            ax2.fill_between(steps[safe_mask], 0, barrier[safe_mask],
                             alpha=0.3, color='green', label='Safe (h ≥ 0)')
        if np.any(unsafe_mask):
            ax2.fill_between(steps[unsafe_mask], barrier[unsafe_mask], 0,
                             alpha=0.3, color='red', label='Unsafe (h < 0)')

        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Barrier function h(x)')
        ax2.set_title('Safety Margin Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Control inputs
        ax3 = axes[0, 2]
        time = np.arange(len(u_des)) * self.dt
        ax3.plot(time, u_des[:, 0], 'r--', label='Desired velocity (v)')
        ax3.plot(time, u_safe[:, 0], 'b-', linewidth=2, label='Safe velocity (v)')
        ax3.plot(time, u_safe[:, 1], 'g-', linewidth=2, label='Angular velocity (ω)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control inputs')
        ax3.set_title('Control Inputs: Desired vs Safe')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Distance to obstacle
        ax4 = axes[1, 0]
        distances = []
        for state in states:
            dist = np.linalg.norm(state[:2] - self.robot.obstacle_pos)
            distances.append(dist)
        distances = np.array(distances)

        time_dist = np.arange(len(distances)) * self.dt
        ax4.plot(time_dist, distances, 'b-', linewidth=2, label='Distance to obstacle')
        ax4.axhline(y=self.robot.obstacle_radius + self.robot.radius,
                    color='r', linestyle='--', alpha=0.5, label='Collision distance')
        ax4.axhline(y=self.robot.obstacle_radius + self.robot.radius + self.robot.safety_margin,
                    color='orange', linestyle='--', alpha=0.5, label='Safety boundary')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Distance')
        ax4.set_title('Distance to Obstacle Center')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: CBF constraint satisfaction
        ax5 = axes[1, 1]
        constraint_values = []
        required_values = []

        for i, state in enumerate(states):
            if i < len(u_safe):
                h_val = barrier[i]
                dh_dx = self.robot.barrier_gradient(state)
                cos_theta = np.cos(state[2])
                sin_theta = np.sin(state[2])
                L_g_h = dh_dx[0] * cos_theta + dh_dx[1] * sin_theta
                constraint_val = L_g_h * u_safe[i, 0]
                required = -self.robot.gamma * h_val
                constraint_values.append(constraint_val)
                required_values.append(required)

        if constraint_values:
            time_constraint = np.arange(len(constraint_values)) * self.dt
            ax5.plot(time_constraint, constraint_values, 'b-', label='L_g h * u')
            ax5.plot(time_constraint, required_values, 'r--', label='-γh (required)')

            # Fill between where constraint is satisfied
            constraint_arr = np.array(constraint_values)
            required_arr = np.array(required_values)
            satisfied_mask = constraint_arr >= required_arr

            if np.any(satisfied_mask):
                ax5.fill_between(time_constraint[satisfied_mask],
                                 constraint_arr[satisfied_mask],
                                 required_arr[satisfied_mask],
                                 alpha=0.3, color='green', label='Constraint satisfied')

            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('CBF constraint value')
            ax5.set_title('CBF Constraint Satisfaction')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No constraint data',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax5.transAxes)
            ax5.set_title('CBF Constraint Satisfaction')

        # Plot 6: Information table
        ax6 = axes[1, 2]
        ax6.axis('off')

        # Calculate min distance
        if len(distances) > 0:
            min_distance = distances.min()
        else:
            min_distance = float('inf')

        info_text = (
            f"CBF Safety Controller Demo\n"
            f"==========================\n"
            f"Robot start: ({states[0, 0]:.1f}, {states[0, 1]:.1f})\n"
            f"Robot end: ({states[-1, 0]:.1f}, {states[-1, 1]:.1f})\n"
            f"Min barrier value: {barrier.min():.3f}\n"
            f"Min distance to obstacle: {min_distance:.3f}\n"
            f"\nParameters:\n"
            f"Safety margin: {self.robot.safety_margin}\n"
            f"CBF γ: {self.robot.gamma}\n"
            f"Max velocity: {self.robot.v_max}\n"
            f"Obstacle radius: {self.robot.obstacle_radius}\n"
            f"Robot radius: {self.robot.radius}\n"
        )

        # Add collision warning if applicable
        if len(barrier) > 0 and barrier.min() < 0:
            info_text += f"\n⚠️  WARNING: Safety violation!\n"
            info_text += f"Minimum h = {barrier.min():.3f}"
        else:
            info_text += f"\n✅ Safety maintained throughout"

        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

        return fig

    def create_animation(self):
        """Create an animation of the simulation"""
        states = np.array(self.history['states'])

        if len(states) == 0:
            print("No data to animate!")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Setup plot
        ax.set_xlim(-1, 7)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('CBF-Controlled Robot Avoiding Obstacle')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Draw obstacle and safety boundary
        obstacle = plt.Circle((self.robot.obstacle_pos[0],
                               self.robot.obstacle_pos[1]),
                              self.robot.obstacle_radius,
                              color='red', alpha=0.5, label='Obstacle')
        ax.add_patch(obstacle)

        safety_boundary = plt.Circle((self.robot.obstacle_pos[0],
                                      self.robot.obstacle_pos[1]),
                                     self.robot.obstacle_radius +
                                     self.robot.radius +
                                     self.robot.safety_margin,
                                     color='orange', alpha=0.2,
                                     linestyle='--', fill=False,
                                     label='Safety boundary')
        ax.add_patch(safety_boundary)

        # Initialize robot patch
        robot_patch = plt.Circle((states[0, 0], states[0, 1]),
                                 self.robot.radius,
                                 color='blue', alpha=0.5)
        ax.add_patch(robot_patch)

        # Initialize heading arrow
        arrow_length = self.robot.radius * 1.5
        heading_arrow = ax.arrow(states[0, 0], states[0, 1],
                                 arrow_length * np.cos(states[0, 2]),
                                 arrow_length * np.sin(states[0, 2]),
                                 head_width=0.1, head_length=0.15,
                                 fc='darkblue', ec='darkblue')

        # Initialize trajectory line
        trajectory_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)

        # Text for status
        status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                              verticalalignment='top')

        # Legend
        ax.legend()

        def update(frame):
            """Update function for animation"""
            # Clear previous arrow
            heading_arrow.remove()

            # Update robot position
            robot_patch.center = (states[frame, 0], states[frame, 1])

            # Update heading arrow
            arrow_length = self.robot.radius * 1.5
            new_arrow = ax.arrow(states[frame, 0], states[frame, 1],
                                 arrow_length * np.cos(states[frame, 2]),
                                 arrow_length * np.sin(states[frame, 2]),
                                 head_width=0.1, head_length=0.15,
                                 fc='darkblue', ec='darkblue')

            # Update trajectory
            trajectory_line.set_data(states[:frame + 1, 0], states[:frame + 1, 1])

            # Update status text
            barrier_val = self.history['barrier'][frame]
            status_text.set_text(f'Step: {frame}\n'
                                 f'Position: ({states[frame, 0]:.2f}, {states[frame, 1]:.2f})\n'
                                 f'Barrier h(x): {barrier_val:.3f}\n'
                                 f'Status: {"SAFE" if barrier_val >= 0 else "UNSAFE"}')

            return robot_patch, new_arrow, trajectory_line, status_text

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(states),
                             interval=100, blit=False, repeat=False)

        plt.tight_layout()
        return anim


def main():
    """Run the simulation and display results"""
    # Create and run simulator
    simulator = CBFSimulator()

    print("=" * 60)
    print("DIFFERENTIAL DRIVE ROBOT WITH CBF SAFETY CONTROL")
    print("=" * 60)
    print("\nScenario: Robot heading straight toward an obstacle")
    print("CBF will slow down the robot to maintain safe distance")

    num_steps = simulator.run_simulation()

    # Plot results
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    simulator.plot_results()

    # Ask about animation
    print("\n" + "=" * 60)
    response = input("Would you like to see an animation? (y/n): ")
    if response.lower() == 'y':
        print("Creating animation...")
        anim = simulator.create_animation()

        if anim:
            # Save animation
            save_response = input("Save animation as GIF? (y/n): ")
            if save_response.lower() == 'y':
                try:
                    anim.save('cbf_simulation.gif', writer='pillow', fps=10)
                    print("Animation saved as 'cbf_simulation.gif'")
                except Exception as e:
                    print(f"Could not save GIF: {e}")

            plt.show()

    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print("\nWhat happened:")
    print("1. Robot wanted to go straight ahead at max speed (unsafe)")
    print("2. CBF safety filter detected approaching obstacle")
    print("3. CBF reduced forward velocity to maintain safe distance")
    print("4. Robot slowed down near obstacle instead of colliding")
    print("\nKey CBF concepts demonstrated:")
    print("- Barrier function h(x) defines safe region (h ≥ 0)")
    print("- CBF constraint: L_f h + L_g h * u ≥ -γh(x)")
    print("- Safety filter modifies desired control when needed")
    print("- Robot maintains safe distance while trying to follow desired path")


if __name__ == "__main__":
    main()