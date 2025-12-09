import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Robot configuration
robot = {
    'x': 0.0,
    'y': 0.0,
    'radius': 0.2,
    'theta': np.pi / 4  # 45 degrees
}

# Small obstacle at top
obstacle = {
    'x': 0.0,
    'y': 2.0,
    'radius': 0.3
}

# LIDAR configuration
num_rays = 72
max_range = 3.0


def calculate_lidar_rays(robot, obstacle, num_rays, max_range):
    """Calculate LIDAR ray intersections with obstacle"""
    rays = []

    for i in range(num_rays):
        # Ray angle in global frame
        ray_angle = 2 * np.pi * i / num_rays

        # Ray direction
        ray_dx = np.cos(ray_angle)
        ray_dy = np.sin(ray_angle)

        # Check intersection with obstacle (circular)
        to_obs_x = obstacle['x'] - robot['x']
        to_obs_y = obstacle['y'] - robot['y']

        a = ray_dx ** 2 + ray_dy ** 2
        b = -2 * (ray_dx * to_obs_x + ray_dy * to_obs_y)
        c = to_obs_x ** 2 + to_obs_y ** 2 - obstacle['radius'] ** 2

        discriminant = b ** 2 - 4 * a * c

        hit_point = None
        distance = max_range
        hit_obstacle = False

        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)
            t = t1 if t1 > 0 else t2

            if t > 0 and t < max_range:
                distance = t
                hit_point = (robot['x'] + t * ray_dx, robot['y'] + t * ray_dy)
                hit_obstacle = True

        # If no hit, ray goes to max range
        if hit_point is None:
            hit_point = (robot['x'] + max_range * ray_dx,
                         robot['y'] + max_range * ray_dy)

        rays.append({
            'angle': ray_angle,
            'hit_point': hit_point,
            'distance': distance,
            'hit_obstacle': hit_obstacle
        })

    return rays


# Calculate all rays
rays = calculate_lidar_rays(robot, obstacle, num_rays, max_range)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_title('LIDAR Ray Visualization - Obstacle Detection', fontsize=14, fontweight='bold')

# Draw max range circle (dashed)
max_range_circle = Circle((robot['x'], robot['y']), max_range,
                          fill=False, edgecolor='gray', linestyle='--',
                          linewidth=1, alpha=0.5, label='Max Range')
ax.add_patch(max_range_circle)

# Draw all LIDAR rays
rays_hit = []
rays_miss = []

for ray in rays:
    if ray['hit_obstacle']:
        # Red rays that hit obstacle
        ax.plot([robot['x'], ray['hit_point'][0]],
                [robot['y'], ray['hit_point'][1]],
                'r-', alpha=0.6, linewidth=1.5)
        # Mark hit points
        ax.plot(ray['hit_point'][0], ray['hit_point'][1],
                'ro', markersize=3, alpha=0.8)
        rays_hit.append(ray)
    else:
        # Light blue rays that miss
        ax.plot([robot['x'], ray['hit_point'][0]],
                [robot['y'], ray['hit_point'][1]],
                'lightblue', alpha=0.2, linewidth=0.5)
        rays_miss.append(ray)

# Draw obstacle
obstacle_circle = Circle((obstacle['x'], obstacle['y']), obstacle['radius'],
                         color='orange', alpha=0.7, edgecolor='darkorange',
                         linewidth=2, label='Obstacle')
ax.add_patch(obstacle_circle)
ax.text(obstacle['x'], obstacle['y'], 'Obstacle',
        ha='center', va='center', fontsize=10, fontweight='bold')

# Draw robot
robot_circle = Circle((robot['x'], robot['y']), robot['radius'],
                      color='blue', alpha=0.8, edgecolor='darkblue',
                      linewidth=2, label='Robot')
ax.add_patch(robot_circle)

# Draw robot direction indicator
dir_length = 0.3
dir_x = robot['x'] + dir_length * np.cos(robot['theta'])
dir_y = robot['y'] + dir_length * np.sin(robot['theta'])
ax.arrow(robot['x'], robot['y'],
         dir_length * np.cos(robot['theta']),
         dir_length * np.sin(robot['theta']),
         head_width=0.1, head_length=0.08, fc='darkblue', ec='darkblue')

ax.text(robot['x'], robot['y'] - 0.4, 'Robot',
        ha='center', va='top', fontsize=10, fontweight='bold', color='blue')

# Add custom legend entries for rays
from matplotlib.lines import Line2D

custom_lines = [
    robot_circle,
    obstacle_circle,
    max_range_circle,
    Line2D([0], [0], color='red', linewidth=2, label='Ray Hit'),
    Line2D([0], [0], color='lightblue', linewidth=2, alpha=0.5, label='Ray Miss')
]
ax.legend(handles=custom_lines, loc='upper right', fontsize=10)

# Add statistics text box
stats_text = f"""LIDAR Statistics:
━━━━━━━━━━━━━━━━
Total Rays: {num_rays}
Rays Hit: {len(rays_hit)}
Rays Miss: {len(rays_miss)}
Max Range: {max_range:.1f}m
━━━━━━━━━━━━━━━━
Robot: ({robot['x']:.1f}, {robot['y']:.1f})
Obstacle: ({obstacle['x']:.1f}, {obstacle['y']:.1f})"""

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

plt.tight_layout()
plt.show()

print(f"\n{'=' * 50}")
print(f"LIDAR SCAN RESULTS")
print(f"{'=' * 50}")
print(f"Total rays emitted: {num_rays}")
print(f"Rays hitting obstacle: {len(rays_hit)}")
print(f"Rays missing obstacle: {len(rays_miss)}")
print(f"Detection rate: {len(rays_hit) / num_rays * 100:.1f}%")
print(f"\nClosest hit point distance: {min([r['distance'] for r in rays_hit]):.3f}m")
print(f"Farthest hit point distance: {max([r['distance'] for r in rays_hit]):.3f}m")
print(f"{'=' * 50}")