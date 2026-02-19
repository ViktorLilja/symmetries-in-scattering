import numpy as np
import matplotlib.pyplot as plt

def visualize(ax, pattern):
    """
    Given the axis of a matplotlib figure and an array of design parameters,
    plots an illustration of the grating unit cell corresponding to the design 
    parameters on the axis.
    """
    pattern = np.array(pattern)
    x = np.linspace(0, len(pattern), len(pattern)+1).repeat(2)/len(pattern)

    # Height matching proportions of grating used in the simulation
    y = np.pad(pattern.repeat(2), pad_width=1, constant_values=False) * 0.918/2.192 

    # Grating
    ax.plot(x, y, "k")
    ax.fill_between(x, 0, y, color="gray") 

    # Substrate
    substrate_height = 0.3
    ax.fill_between([0, 0, 1, 1], -substrate_height, [-substrate_height, 0, 0, -substrate_height], color="lightgray")
    ax.plot([0, 0, 1, 1], [-substrate_height, 0, 0, -substrate_height], "k")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)