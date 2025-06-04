from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt

L = 0.0319
R = 0.005
l = 0.006

Lx = L * np.cos(np.pi / 6)
Ly = L * np.sin(np.pi / 6)

@dataclass
class Spinner:
    number: int
    theta: float
    x: float
    y: float

@dataclass
class OR:
    A: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=0.0, y=0.0))
    B: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=2 * Lx, y=0))
    F: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=Lx, y=-Ly))
    N: int = 3

@dataclass
class AND:
    A: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=0.0, y=0.0))
    B: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=2 * Lx, y=0))
    F: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=Lx, y=Ly))
    N: int = 3

@dataclass
class NOT:
    A: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=0.0, y=0.0))
    F: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=0, y=-L))
    N: int = 2


def plot_spinner(spinner: Spinner, ax=None, radius=0.005, branch_length=0.01, color='blue'):
    if ax is None:
        fig, ax = plt.subplots()
    
    x0, y0 = spinner.x, spinner.y
    theta = spinner.theta
    
    # Cercle
    circle = plt.Circle((x0, y0), radius, fill=False, color=color)
    ax.add_patch(circle)
    
    # 3 branches espacées de 120°
    angles = [theta + i * 2 * np.pi / 3 for i in range(3)]
    for angle in angles:
        x_end = x0 + branch_length * np.cos(angle)
        y_end = y0 + branch_length * np.sin(angle)
        ax.plot([x0, x_end], [y0, y_end], color=color, linewidth=2)

    ax.set_aspect('equal')
    ax.autoscale(enable=True)
    ax.grid(True)

def plot_and_gate(and_gate: AND):
    fig, ax = plt.subplots()
    # Affiche chaque spinner de la porte AND en couleur différente
    plot_spinner(and_gate.A, ax, color='red')
    plot_spinner(and_gate.B, ax, color='green')
    plot_spinner(and_gate.F, ax, color='blue')
    
    ax.set_title("Porte AND avec spinners")
    plt.show()

# Exemple d'utilisation
and_gate = AND()
plot_and_gate(and_gate)



