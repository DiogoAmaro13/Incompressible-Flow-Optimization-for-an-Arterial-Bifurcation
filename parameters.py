import numpy as np

# ====================================================== GEOMETRICAL PARAMETERS ====================================================== #
mesh_size: float = 0.05
mu1: float = 1.0
mu2: float = np.pi / 5.0
mu3: float = np.pi / 6.0
mu4: float = 1.0
mu5: float = 1.7
mu6: float = 2.2

if mesh_size <= 0:
    raise ValueError("mesh_size should be positive")

Y: float = 1.0
X: float = -Y
L: float = 3.0
B: float = Y - mu1
H_1: float = B + np.tan(mu2) * mu5
H_2: float = B - np.tan(mu3) * mu6
L_1: float = mu1 * np.cos(mu2) * np.sin(mu2)
L_2: float = (B - X) * np.cos(mu3) * np.sin(mu3)
N: float = mu1 * np.cos(mu2) * np.cos(mu2)
M: float = - (B - X) * np.cos(mu3) * np.cos(mu3)

# ====================================================== TAGS ====================================================== #
inlet_tag: int = 1
wall_tag: int = 2
control_tag: int = 3
obs_tag: int = 4

# ====================================================== PROBLEM DATA ====================================================== #
nu: float = 0.04                          # viscosity
reg_smooth: float = 0.001
reg_l2: float = 0.1 * reg_smooth          # Tikhonov reg on the control
mixing_ratio: float = 0.8