import numpy as np
import matplotlib.pyplot as plt
import os

pwd = os.path.dirname(os.path.abspath(__file__))

# lp_norm:
# ||w||_p = (\sum_i^d |w_i|^p)^(1 / p)
def lp_norm(w, p):
    return np.sum(np.abs(w) ** p, axis=0) ** (1 / p)

# Plot isocontours of lp norms for p = 1
w1 = np.linspace(-3, 3, 600)
w2 = np.linspace(-3, 3, 600)
W1, W2 = np.meshgrid(w1, w2)
Z1 = lp_norm(np.array([W1, W2]), 1)
plt.figure(figsize=(6, 6))
plt.contour(W1, W2, Z1, cmap='viridis', levels=10)
plt.title('Isocontours of $l_1$ norm')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.axis('equal')
plt.savefig(os.path.join(pwd, 'q5_lp_norm_1.png'))

# Plot isocontours of lp norms for p = 2
Z2 = lp_norm(np.array([W1, W2]), 2)
plt.figure(figsize=(6, 6))
plt.contour(W1, W2, Z2, cmap='viridis', levels=10)
plt.title('Isocontours of $l_2$ norm')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.axis('equal')
plt.savefig(os.path.join(pwd, 'q5_lp_norm_2.png'))

# Plot isocontours of lp norms for p = 0.5
Z0_5 = lp_norm(np.array([W1, W2]), 0.5)
plt.figure(figsize=(6, 6))
plt.contour(W1, W2, Z0_5, cmap='viridis', levels=10)
plt.title('Isocontours of $l_{0.5}$ norm')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.axis('equal')
plt.savefig(os.path.join(pwd, 'q5_lp_norm_0p5.png'))