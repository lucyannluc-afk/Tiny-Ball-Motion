import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Environement:
  def __init__(self, grid_size, domain_size, num_obstacles, obstacle_size, num_particles
