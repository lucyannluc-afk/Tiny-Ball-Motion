import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Environement:
  def __init__(self, grid_size : int = 200, domain_size : int = 25,  num_obstacles : int = 5,  obstacle_size : size = 20,  num_particles : size = 20, seed : int = 46):
     self.grid_size = grid_size
     self.domain_size = domain_size
     self.num_obstacles = num_obstacles
     self.obstacle_size = obstacle_size
        
     np.random.seed(seed)

def
