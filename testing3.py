import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize


class TurtleBot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.radius = 0.2  # Robot radius
        self.max_v = 0.5  # Max linear velocity (m/s)
        self.max_w = 1.5  # Max angular velocity (rad/s)
        self.max_a = 0.5  # Max linear acceleration (m/s^2)
        self.max_alpha = 2.0  # Max angular acceleration (rad/s^2)

        # Current velocities
        self.v = 0.0
        self.w = 0.0

        # History storage
        self.history = {
            'x': [x],
            'y': [y],
            'theta': [theta],
            'cbf_active': [False],
            'h_values': [0.0],
            'v': [0.0],
            'w': [0.0],
            'time': [0.0]
        }

    def apply_acceleration_limits(self, v_cmd, w_cmd, dt):
        """Apply acceleration limits to commanded velocities"""
        # Linear acceleration limit
        dv = v_cmd - self.v
        max_dv = self.max_a * dt
        if abs(dv) > max_dv:
            dv = np.sign(dv) * max_dv
        v_limited = self.v + dv

        # Angular acceleration limit
        dw = w_cmd - self.w
        max_dw = self.max_alpha * dt
        if abs(dw) > max_dw:
            dw = np.sign(dw) * max_dw
        w_limited = self.w + dw

        # Also enforce velocity limits
        v_limited = np.clip(v_limited, 0, self.max_v)
        w_limited = np.clip(w_limited, -self.max_w, self.max_w)

        return v_limited, w_limited

    def update(self, v_cmd, w_cmd, dt, time, cbf_active, h_value):
        """Update robot state with velocity commands and log all data"""
        # Apply acceleration limits
        v_actual, w_actual = self.apply_acceleration_limits(v_cmd, w_cmd, dt)

        # Update velocities
        self.v = v_actual
        self.w = w_actual

        # Update position
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.w * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # Normalize

        # Store everything together
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['theta'].append(self.theta)
        self.history['v'].append(self.v)
        self.history['w'].append(self.w)
        self.history['time'].append(time)
        self.history['cbf_active'].append(cbf_active)
        self.history['h_values'].append(h_value)


class CBFController:
    def __init__(self, gamma=3.0):
        self.gamma = gamma

    def nominal_controller(self, robot, goal):
        """Simple proportional controller to drive to goal"""
        dx = goal[0] - robot.x
        dy = goal[1] - robot.y

        # Desired velocity
        distance = np.sqrt(dx ** 2 + dy ** 2)
        v_des = min(0.4, 0.5 * distance)

        # Desired angular velocity (proportional to angle error)
        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(angle_to_goal - robot.theta),
                                np.cos(angle_to_goal - robot.theta))
        w_des = 3.0 * angle_diff

        return v_des, w_des

    def compute_cbf(self, robot, obstacle):
        """Compute CBF value h(x) = ||p - p_obs||^2 - (R + R_w)^2"""
        dx = robot.x - obstacle['x']
        dy = robot.y - obstacle['y']
        dist_sq = dx ** 2 + dy ** 2
        safe_dist = robot.radius + obstacle['radius']
        h = dist_sq - safe_dist ** 2
        return h, dx, dy

    def compute_cbf_derivative(self, robot, obstacle, dx, dy, v, w):
        """
        Compute CBF time derivative
        h(x) = ||p - p_obs||^2 - r^2
        h_dot = 2*(p - p_obs)^T * p_dot
        where p_dot = [v*cos(theta), v*sin(theta)]^T
        """
        h_dot = 2 * (dx * v * np.cos(robot.theta) + dy * v * np.sin(robot.theta))
        return h_dot

    def safe_controller(self, robot, goal, obstacle):
        """CBF-based QP controller with turning capability"""
        v_des, w_des = self.nominal_controller(robot, goal)

        # Compute CBF value
        h, dx, dy = self.compute_cbf(robot, obstacle)

        # Only activate CBF if close to obstacle
        if h > 2.0:
            return v_des, w_des, False  # Not close, use nominal control

        # Check if we're heading toward the obstacle
        angle_to_obs = np.arctan2(-dy, -dx)  # Angle from robot to obstacle
        angle_diff = np.arctan2(np.sin(angle_to_obs - robot.theta),
                                np.cos(angle_to_obs - robot.theta))

        # If pointing toward obstacle and close, we need to turn
        heading_toward_obstacle = abs(angle_diff) < np.pi / 3 and h < 1.0

        if heading_toward_obstacle:
            # Force turning by heavily penalizing forward motion
            # and encouraging turning away
            turn_direction = np.sign(angle_diff) if angle_diff != 0 else 1.0
            # Reverse the turn direction to go away from obstacle
            turn_direction = -turn_direction

            # Define QP with modified objective that encourages turning
            def objective(u):
                v, w = u
                # Heavy penalty on forward velocity when heading toward obstacle
                v_penalty = 10.0 * v ** 2 if h < 0.5 else 5.0 * (v - v_des) ** 2
                # Encourage turning away from obstacle
                w_penalty = (w - turn_direction * robot.max_w * 0.7) ** 2
                return v_penalty + 0.3 * w_penalty
        else:
            # Normal QP objective
            def objective(u):
                v, w = u
                return (v - v_des) ** 2 + 0.5 * (w - w_des) ** 2

        def constraint(u):
            v, w = u
            h_dot = self.compute_cbf_derivative(robot, obstacle, dx, dy, v, w)
            return h_dot + self.gamma * h

        # Bounds on controls
        bounds = [(0, robot.max_v), (-robot.max_w, robot.max_w)]

        # Constraint: must be >= 0
        cons = {'type': 'ineq', 'fun': constraint}

        # Initial guess: if heading toward obstacle, suggest turning
        if heading_toward_obstacle:
            turn_direction = -np.sign(angle_diff) if angle_diff != 0 else 1.0
            initial_guess = [0.1, turn_direction * robot.max_w * 0.5]
        else:
            initial_guess = [v_des, w_des]

        # Solve QP
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        if result.success:
            return result.x[0], result.x[1], True
        else:
            # Emergency: turn away from obstacle
            turn_direction = -np.sign(angle_diff) if angle_diff != 0 else 1.0
            print(f"QP failed at h={h:.3f}, turning away")
            return 0.0, turn_direction * robot.max_w, True


def run_simulation():
    # Initialize robot
    robot = TurtleBot(0.0, 0.0, 0.0)
    goal = np.array([5.0, 3.0])

    # Single obstacle - moved up and left a bit
    obstacle = {'x': 2.2, 'y': 1.7, 'radius': 0.5}

    # Simulation parameters
    dt = 0.05
    max_time = 60.0
    time = 0.0

    controller = CBFController(gamma=3.0)

    # Run simulation
    print("Starting simulation...")
    print(f"Robot max acceleration: {robot.max_a} m/s²")
    print(f"Robot max angular acceleration: {robot.max_alpha} rad/s²")

    while time < max_time:
        # Check if goal reached
        dist_to_goal = np.sqrt((robot.x - goal[0]) ** 2 + (robot.y - goal[1]) ** 2)
        if dist_to_goal < 0.15:
            print(f"Goal reached at t={time:.2f}s")
            break

        # Compute safe control
        v, w, cbf_active = controller.safe_controller(robot, goal, obstacle)

        # Compute CBF value for logging
        h, _, _ = controller.compute_cbf(robot, obstacle)

        # Update robot with all data at once
        robot.update(v, w, dt, time, cbf_active, h)

        time += dt

    print(f"Simulation completed at t={time:.2f}s")
    print(f"Minimum CBF value: {min(robot.history['h_values']):.3f}")

    return robot, obstacle, goal


# Run simulation
robot, obstacle, goal = run_simulation()

# Get trajectory from robot's history
trajectory = robot.history

# Create figure with four subplots
fig = plt.figure(figsize=(16, 10))

# Top left: Trajectory
ax1 = fig.add_subplot(221)
ax1.set_xlim(-0.5, 6)
ax1.set_ylim(-0.5, 4)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_title('TurtleBot Trajectory with CBF Safety', fontsize=14, fontweight='bold')

# Plot obstacle
circle = Circle((obstacle['x'], obstacle['y']), obstacle['radius'],
                color='red', alpha=0.7, label='Obstacle')
ax1.add_patch(circle)

# Safety boundary
safety_circle = Circle((obstacle['x'], obstacle['y']),
                       obstacle['radius'] + robot.radius,
                       color='red', alpha=0.2, linestyle='--',
                       fill=False, linewidth=2, label='Safety Boundary')
ax1.add_patch(safety_circle)

# Color trajectory by CBF activation
for i in range(len(trajectory['x']) - 1):
    color = 'orange' if trajectory['cbf_active'][i] else 'blue'
    ax1.plot(trajectory['x'][i:i + 2], trajectory['y'][i:i + 2],
             color=color, linewidth=2, alpha=0.8)

# Add legend elements
ax1.plot([], [], 'b-', linewidth=2, label='Nominal Control')
ax1.plot([], [], color='orange', linewidth=2, label='CBF Active')

# Plot goal
ax1.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')

# Plot start and end positions
ax1.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=12, label='Start')
ax1.plot(trajectory['x'][-1], trajectory['y'][-1], 'gs', markersize=12, label='End')

# Plot robot orientation at intervals
step = max(1, len(trajectory['x']) // 20)
for i in range(0, len(trajectory['x']), step):
    x, y, theta = trajectory['x'][i], trajectory['y'][i], trajectory['theta'][i]
    dx = 0.25 * np.cos(theta)
    dy = 0.25 * np.sin(theta)
    ax1.arrow(x, y, dx, dy, head_width=0.12, head_length=0.08,
              fc='darkblue', ec='darkblue', alpha=0.6)

ax1.legend(loc='upper left', fontsize=9)

# Top right: CBF value over time
ax2 = fig.add_subplot(222)
time_array = np.array(trajectory['time'])
ax2.plot(time_array, trajectory['h_values'], 'b-', linewidth=2, label='h(x)')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Safety Boundary (h=0)')
ax2.fill_between(time_array, -1, 0, alpha=0.2, color='red', label='Unsafe Region')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('CBF Value h(x)', fontsize=12)
ax2.set_title('Control Barrier Function Over Time', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(-0.5, max(trajectory['h_values']) + 0.5)

# Bottom left: Linear velocity over time
ax3 = fig.add_subplot(223)
ax3.plot(time_array, trajectory['v'], 'g-', linewidth=2, label='Linear velocity (v)')
ax3.axhline(y=robot.max_v, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Max velocity')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Linear Velocity (m/s)', fontsize=12)
ax3.set_title('Linear Velocity Over Time', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.set_ylim(-0.1, robot.max_v + 0.1)

# Highlight when CBF is active
for i in range(len(time_array) - 1):
    if trajectory['cbf_active'][i]:
        ax3.axvspan(time_array[i], time_array[i + 1], alpha=0.2, color='orange')

# Bottom right: Angular velocity over time
ax4 = fig.add_subplot(224)
ax4.plot(time_array, trajectory['w'], 'purple', linewidth=2, label='Angular velocity (w)')
ax4.axhline(y=robot.max_w, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Max velocity')
ax4.axhline(y=-robot.max_w, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
ax4.set_title('Angular Velocity Over Time', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.set_ylim(-robot.max_w - 0.2, robot.max_w + 0.2)

# Highlight when CBF is active
for i in range(len(time_array) - 1):
    if trajectory['cbf_active'][i]:
        ax4.axvspan(time_array[i], time_array[i + 1], alpha=0.2, color='orange')

# Add text box with info
textstr = f'Total waypoints: {len(trajectory["x"])}\n'
textstr += f'γ (gamma): 3.0\n'
textstr += f'Max accel: {robot.max_a} m/s²\n'
textstr += f'Max ang accel: {robot.max_alpha} rad/s²\n'
textstr += f'Min h(x): {min(trajectory["h_values"]):.3f}\n'
textstr += f'CBF active: {sum(trajectory["cbf_active"])}/{len(trajectory["cbf_active"])} steps\n'
textstr += f'Safety maintained: {"✓" if min(trajectory["h_values"]) >= 0 else "✗"}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.show()

print(f"\n{'=' * 50}")
print(f"SIMULATION RESULTS")
print(f"{'=' * 50}")
print(f"Total trajectory points: {len(trajectory['x'])}")
print(f"Start position: ({trajectory['x'][0]:.2f}, {trajectory['y'][0]:.2f})")
print(f"End position: ({trajectory['x'][-1]:.2f}, {trajectory['y'][-1]:.2f})")
print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
print(f"Minimum CBF value (h_min): {min(trajectory['h_values']):.4f}")
print(f"Safety maintained: {min(trajectory['h_values']) >= 0}")
print(f"CBF was active for {sum(trajectory['cbf_active'])} time steps")
print(f"Robot traveled from history: {len(robot.history['x'])} points")
print(f"Average linear velocity: {np.mean(trajectory['v']):.3f} m/s")
print(f"Max linear velocity: {np.max(trajectory['v']):.3f} m/s")
print(f"Min linear velocity: {np.min(trajectory['v']):.3f} m/s")
print(f"Average angular velocity magnitude: {np.mean(np.abs(trajectory['w'])):.3f} rad/s")
print(f"{'=' * 50}")