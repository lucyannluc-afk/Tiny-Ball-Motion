import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parameters
grid_size = 200
domain_size = 25
num_obstacles = 5
obstacle_size = 20
num_particles = 20
dt = 0.01
T = 20
N = int(T / dt)
r = 0.01
m = 1.0

# Seed for reproducibility
np.random.seed(46)

# 1. Generate viscosity grid
raw_viscosity = np.random.rand(grid_size, grid_size)
viscosity_grid = gaussian_filter(raw_viscosity, sigma=8)
viscosity_grid = (viscosity_grid - viscosity_grid.min()) / (viscosity_grid.max() - viscosity_grid.min()) * 2.5 + 0.5

# 2. Generate obstacles and record physical bounds (using pixel edge alignment)
obstacle_grid = np.zeros((grid_size, grid_size))
obstacle_bounds = []

def grid_to_phys_edge(i, grid_size, domain_size):
    return i / grid_size * domain_size - domain_size / 2

for _ in range(num_obstacles):
    x_start = np.random.randint(10, grid_size - obstacle_size - 10)
    y_start = np.random.randint(10, grid_size - obstacle_size - 10)
    obstacle_grid[x_start:x_start + obstacle_size, y_start:y_start + obstacle_size] = 1

    x0 = grid_to_phys_edge(x_start, grid_size, domain_size)
    x1 = grid_to_phys_edge(x_start + obstacle_size, grid_size, domain_size)
    y0 = grid_to_phys_edge(y_start, grid_size, domain_size)
    y1 = grid_to_phys_edge(y_start + obstacle_size, grid_size, domain_size)
    obstacle_bounds.append((x0, x1, y0, y1))

# 3. Mask obstacle region in all physical quantities
#viscosity_grid[obstacle_grid == 1] = 1e6
D_grid = 1.0 / viscosity_grid
D_grid[obstacle_grid == 1] = 0.0
force_grid_x = np.random.uniform(-0.05, 0.05, (grid_size, grid_size))
force_grid_y = np.random.uniform(-0.05, 0.05, (grid_size, grid_size))
force_grid_x[obstacle_grid == 1] = 0.0
force_grid_y[obstacle_grid == 1] = 0.0

# 4. Initialize particle arrays
x = np.zeros((num_particles, N))
y = np.zeros((num_particles, N))
vx = np.zeros((num_particles, N))
vy = np.zeros((num_particles, N))
eta_x = np.random.normal(0, 1, (num_particles, N))
eta_y = np.random.normal(0, 1, (num_particles, N))

# Find all valid (non-obstacle) grid positions
valid_indices = [(gx, gy) for gx in range(grid_size) for gy in range(grid_size) if obstacle_grid[gx, gy] == 0]

def grid_to_phys_center(g, grid_size, domain_size):
    return (g + 0.5) / grid_size * domain_size - domain_size / 2

def is_in_obstacle(x, y, bounds, margin=0.01):
    for (x0, x1, y0, y1) in bounds:
        if (x0 - margin <= x <= x1 + margin) and (y0 - margin <= y <= y1 + margin):
            return True
    return False

# Initialize particle positions
for i in range(num_particles):
    gx, gy = valid_indices[np.random.randint(len(valid_indices))]
    x[i, 0] = grid_to_phys_center(gx, grid_size, domain_size)
    y[i, 0] = grid_to_phys_center(gy, grid_size, domain_size)

# 5. Langevin simulation with physical-bound obstacle rejection
for n in range(N - 1):
    for i in range(num_particles):
        gx = int((x[i, n] + domain_size / 2) / domain_size * grid_size)
        gy = int((y[i, n] + domain_size / 2) / domain_size * grid_size)
        gx = np.clip(gx, 0, grid_size - 1)
        gy = np.clip(gy, 0, grid_size - 1)

        D = D_grid[gx, gy]
        beta = (6 * np.pi * viscosity_grid[gx, gy] * r) / m
        Fx = force_grid_x[gx, gy]
        Fy = force_grid_y[gx, gy]

        vx_new = vx[i, n] - beta * vx[i, n] * dt + np.sqrt(2 * D * dt) * eta_x[i, n] + Fx * dt / m
        vy_new = vy[i, n] - beta * vy[i, n] * dt + np.sqrt(2 * D * dt) * eta_y[i, n] + Fy * dt / m

        x_new = x[i, n] + vx_new * dt
        y_new = y[i, n] + vy_new * dt

        if (-domain_size / 2 <= x_new <= domain_size / 2 and
            -domain_size / 2 <= y_new <= domain_size / 2 and
            not is_in_obstacle(x_new, y_new, obstacle_bounds, margin=r)):
            x[i, n+1] = x_new
            y[i, n+1] = y_new
            vx[i, n+1] = vx_new
            vy[i, n+1] = vy_new
        else:
            x[i, n+1] = x[i, n]
            y[i, n+1] = y[i, n]
            vx[i, n+1] = 0.0
            vy[i, n+1] = 0.0

# 6. Plot 1: environment only (viscosity + obstacles + colorbar)
plt.figure(figsize=(8, 6))
im = plt.imshow(viscosity_grid, extent=[-12.5, 12.5, -12.5, 12.5], origin="lower", cmap="coolwarm", alpha=0.7)
#plt.imshow(obstacle_grid, extent=[-12.5, 12.5, -12.5, 12.5], origin="lower", cmap="gray", alpha=0.5)
for (x0, x1, y0, y1) in obstacle_bounds:
    plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k-', lw=2)
plt.colorbar(im, label="Viscosity")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cellular Environment")
plt.grid()
plt.tight_layout()
plt.show()

# 7. Plot 2: add particle paths (without scatter points)
plt.figure(figsize=(8, 6))
plt.imshow(viscosity_grid, extent=[-12.5, 12.5, -12.5, 12.5], origin="lower", cmap="coolwarm", alpha=0.7)
#plt.imshow(obstacle_grid, extent=[-12.5, 12.5, -12.5, 12.5], origin="lower", cmap="gray", alpha=0.5)
for (x0, x1, y0, y1) in obstacle_bounds:
    plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k-', lw=2)

for i in range(num_particles):
    plt.plot(x[i, :], y[i, :], linewidth=1, alpha=0.8)

plt.scatter(x[:, 0], y[:, 0], color="red", label="Initial", zorder=3)
plt.scatter(x[:, -1], y[:, -1], color="blue", label="Final", zorder=3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Brownian Movements in Cellular Environment")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
