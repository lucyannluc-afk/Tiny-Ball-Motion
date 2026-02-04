import numpy as np
import tkinter as tk
from tkinter import ttk
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ball import Ball, RectObstacle, World


def generate_obstacles(rng, n_obs, obs_size, half_size, margin=0.2):
   #génération d'obstacles placés aléatoirement
    h = half_size
    s = obs_size
    obstacles = []

    x_min = -h + margin + s / 2 #limite gauche pour le centre
    x_max = h - margin - s / 2 #limite droite pour le centre
    y_min = -h + margin + s / 2
    y_max = h - margin - s / 2

    if x_min >= x_max or y_min >= y_max:
        return []

    for _ in range(n_obs):
        cx = rng.uniform(x_min, x_max) #centre x et y aléatoire, pour chaque obstacle on calcule le centre et on fait un rectangle autour
        cy = rng.uniform(y_min, y_max)
        obstacles.append(RectObstacle(cx - s/2, cx + s/2, cy - s/2, cy + s/2))

    return obstacles


def generate_balls(rng, n_balls, r_ball, half_size, obstacles, v0=2.0, max_tries=5000):
   # on place aléatoirement les balles en évitant les obstacles
    h = half_size
    balls = []

    tries = 0
    while len(balls) < n_balls:
        tries += 1
        if tries > max_tries:
            break #on évite une boucle infinie

        x = rng.uniform(-h + r_ball, h - r_ball) #position aléatoire
        y = rng.uniform(-h + r_ball, h - r_ball)

        # Éviter les obstacles
        ok = True
        for obs in obstacles:
            if (obs.x0 - r_ball <= x <= obs.x1 + r_ball) and (obs.y0 - r_ball <= y <= obs.y1 + r_ball):
                ok = False
                break
        if not ok:
            continue

        # Vitesse initiale aléatoire
        theta = rng.uniform(0, 2 * np.pi)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        balls.append(Ball(x, y, vx, vy, mass=1.0, r=r_ball))

    return balls


class BallSimulationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ball + Obstacles Simulation")

        self.world = None
        self.running = False
        self.dt = 0.02
        self.world_size = 5.0

        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        self.setup_ui()
        self.reset_simulation()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Parameters frame
        params_frame = ttk.LabelFrame(left_panel, text="Paramètres", padding="10")
        params_frame.grid(row=0, column=0, sticky="new")

        # Environment half-size
        ttk.Label(params_frame, text="Taille environnement (H):").grid(row=0, column=0, sticky="w", pady=2)
        self.env_half_var = tk.StringVar(value="5.0")
        ttk.Entry(params_frame, textvariable=self.env_half_var, width=10).grid(row=0, column=1, pady=2)

        # Ball radius
        ttk.Label(params_frame, text="Rayon des balles (r):").grid(row=1, column=0, sticky="w", pady=2)
        self.ball_r_var = tk.StringVar(value="0.3")
        ttk.Entry(params_frame, textvariable=self.ball_r_var, width=10).grid(row=1, column=1, pady=2)

        # Obstacle width
        ttk.Label(params_frame, text="Taille obstacles (w):").grid(row=2, column=0, sticky="w", pady=2)
        self.obs_w_var = tk.StringVar(value="2.0")
        ttk.Entry(params_frame, textvariable=self.obs_w_var, width=10).grid(row=2, column=1, pady=2)

        # Time step
        ttk.Label(params_frame, text="Pas de temps (dt):").grid(row=3, column=0, sticky="w", pady=2)
        self.dt_var = tk.StringVar(value="0.02")
        ttk.Entry(params_frame, textvariable=self.dt_var, width=10).grid(row=3, column=1, pady=2)

        # Number of balls
        ttk.Label(params_frame, text="Nombre de balles:").grid(row=4, column=0, sticky="w", pady=2)
        self.nb_balls_var = tk.StringVar(value="3")
        ttk.Entry(params_frame, textvariable=self.nb_balls_var, width=10).grid(row=4, column=1, pady=2)

        # Number of obstacles
        ttk.Label(params_frame, text="Nombre d'obstacles:").grid(row=5, column=0, sticky="w", pady=2)
        self.nb_obs_var = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.nb_obs_var, width=10).grid(row=5, column=1, pady=2)

        # Options frame
        options_frame = ttk.LabelFrame(left_panel, text="Options", padding="10")
        options_frame.grid(row=1, column=0, sticky="new", pady=10)

        self.show_trails_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Afficher trajectoires",
                        variable=self.show_trails_var).grid(row=0, column=0, sticky="w")

        # Buttons frame
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.grid(row=2, column=0, sticky="new", pady=10)

        self.start_btn = ttk.Button(buttons_frame, text="Start", command=self.toggle_simulation)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.reset_btn = ttk.Button(buttons_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=1, padx=5)

        self.clear_btn = ttk.Button(buttons_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=2, padx=5)

        # Info frame
        info_frame = ttk.LabelFrame(left_panel, text="Infos", padding="10")
        info_frame.grid(row=3, column=0, sticky="new", pady=10)

        self.energy_label = ttk.Label(info_frame, text="Energie totale: 0.00")
        self.energy_label.grid(row=0, column=0, sticky="w")

        self.balls_label = ttk.Label(info_frame, text="Balles: 0")
        self.balls_label.grid(row=1, column=0, sticky="w")

        self.time_label = ttk.Label(info_frame, text="Temps: 0.00")
        self.time_label.grid(row=2, column=0, sticky="w")

        # Right panel - Matplotlib canvas
        self.fig = Figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self.time = 0.0

    def reset_simulation(self):
        """Réinitialise la simulation avec les paramètres actuels"""
        self.running = False
        self.start_btn.config(text="Start")
        self.time = 0.0

        try:
            H = float(self.env_half_var.get())
            r = float(self.ball_r_var.get())
            w = float(self.obs_w_var.get())
            self.dt = float(self.dt_var.get())
            nb_balls = int(float(self.nb_balls_var.get()))
            nb_obs = int(float(self.nb_obs_var.get()))
        except ValueError:
            return

        self.world_size = H

        # Générer obstacles et balles
        rng = np.random.default_rng()
        obstacles = generate_obstacles(rng, nb_obs, w, H)
        self.world = World(half_size=H, obstacle=obstacles)

        balls = generate_balls(rng, nb_balls, r, H, obstacles)
        for ball in balls:
            ball.trail = []
            self.world.add_ball(ball)

        self.draw_static()
        self.draw_balls()
        self.update_info()

    def draw_static(self):
        """Dessine les éléments statiques (limites et obstacles)"""
        self.ax.clear()
        H = self.world_size

        # Limites
        self.ax.plot([-H, H, H, -H, -H], [-H, -H, H, H, -H], 'k-', lw=2)

        # Obstacles
        for obs in self.world.obstacles:
            xs = [obs.x0, obs.x1, obs.x1, obs.x0, obs.x0]
            ys = [obs.y0, obs.y0, obs.y1, obs.y1, obs.y0]
            self.ax.fill(xs, ys, color='#34495e', alpha=0.8)
            self.ax.plot(xs, ys, color='#2c3e50', lw=2)

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-H * 1.05, H * 1.05)
        self.ax.set_ylim(-H * 1.05, H * 1.05)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def draw_balls(self):
    
        # Redessiner les éléments statiques
        self.draw_static()

        for i, ball in enumerate(self.world.balls):
            color = self.colors[i % len(self.colors)]

            # Trajectoire
            if self.show_trails_var.get() and hasattr(ball, 'trail') and len(ball.trail) >= 2:
                trail = np.array(ball.trail)
                self.ax.plot(trail[:, 0], trail[:, 1], color=color, lw=1, alpha=0.6)

            # Balle
            circle = plt.Circle((ball.x, ball.y), ball.r, color=color, ec='black', lw=1.5)
            self.ax.add_patch(circle)

        self.ax.set_title(f"Simulation (t = {self.time:.2f}s)")
        self.canvas.draw()

    def toggle_simulation(self):
        
        self.running = not self.running
        self.start_btn.config(text="Stop" if self.running else "Start")
        if self.running:
            self.animate()

    def animate(self):
       
        if not self.running or self.world is None:
            return

        # Pas de simulation
        self.world.step(self.dt)
        self.time += self.dt

        # Mise à jour des trajectoires
        for ball in self.world.balls:
            if hasattr(ball, 'trail'):
                ball.trail.append((ball.x, ball.y))
                if len(ball.trail) > 200:
                    ball.trail.pop(0)

        self.draw_balls()
        self.update_info()

        # Prochain frame (~30 FPS pour matplotlib)
        self.root.after(33, self.animate)

    def update_info(self):
       
        if self.world:
            total_energy = sum(ball.kinetic_energy for ball in self.world.balls)
            self.energy_label.config(text=f"Energie totale: {total_energy:.2f}")
            self.balls_label.config(text=f"Balles: {len(self.world.balls)}")
            self.time_label.config(text=f"Temps: {self.time:.2f}")

    def clear_canvas(self):
       
        self.ax.clear()
        self.canvas.draw()

    def run(self):
        
        self.root.mainloop()


# Import matplotlib.pyplot pour Circle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    app = BallSimulationGUI()
    app.run()
