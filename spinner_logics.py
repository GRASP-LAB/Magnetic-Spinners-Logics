from dataclasses import dataclass, field, fields
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

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
    A: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi / 6, x=0.0, y=0.0))
    F: Spinner = field(default_factory=lambda: Spinner(number=1, theta=-np.pi / 6, x=0, y=-L))
    N: int = 2

@dataclass
class OAI:
    # OR gate
    A: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=0.0, y=0.0))
    B: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=2 * Lx, y=0))
    AB: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=Lx, y=-Ly))

    # AND gate
    A2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=0.0 + 3 * Lx, y=0.0 - 7 * Ly))
    C: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=2 * Lx + 3 * Lx, y=0 - 7 * Ly))
    B2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=Lx + 3 * Lx, y=Ly - 7 * Ly))

    # cable
    C1 : Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=Lx, y=-3 * Ly))
    C2 : Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=2 * Lx, y=-4 * Ly))
    C3 : Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=2 * Lx, y=-6 * Ly))
    C4 :Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=Lx + 3 * Lx, y=Ly - 5 * Ly))

    # NOT gate
    A3 :Spinner = field(default_factory=lambda: Spinner(number=1, theta=0.0, x=Lx + 4 * Lx, y=Ly - 4 * Ly))
    F :Spinner = field(default_factory=lambda: Spinner(number=-1, theta=0.0, x=Lx + 4 * Lx, y=Ly - 2 * Ly))

    N: int = 0  # temporaire

    def __post_init__(self):
        # N = nombre d'attributs Spinner dans cette classe (ici 2)
        self.N = sum(1 for f in fields(self) if f.type == Spinner)

def U(spinner1: Spinner, spinner2: Spinner) -> float:
    """
    Calcule l'énergie magnétique entre deux spinners selon la formule avec somme sur i,j (simplifiée ici).
    Les angles theta sont en radians.
    """

    energy = 0
    # Paramètres globaux utilisés pour le calcul
    L_ref = L - 2 * (R + l/2)  # normalisation distance
    
    # Angles theta des spinners
    theta1 = spinner1.theta  # en radians
    theta2 = spinner2.theta  # en radians

    # Positions des spinners
    x1, y1 = spinner1.x, spinner1.y
    x2, y2 = spinner2.x, spinner2.y

    # On fait une somme sur i et j dans {0, 2, 4} (équivalent à 0°, 120°, 240°)
    for i in range(3):  # i = 0,1,2 correspond à 0°, 120°, 240°
        for j in range(3):
            # Angles décalés en radians, ici on fait 60° * (theta + i) convertis en radians
            angle1 = theta1 + i * 2 * np.pi / 3
            angle2 = theta2 + j * 2 * np.pi / 3

            # Positions décalées autour de chaque spinner (rayon R+l/2)
            r1 = np.array([
                x1 + (R + l/2) * np.cos(angle1),
                y1 + (R + l/2) * np.sin(angle1)
            ])

            r2 = np.array([
                x2 + (R + l/2) * np.cos(angle2),
                y2 + (R + l/2) * np.sin(angle2)
            ])

            # Vecteurs unitaires selon angles décalés
            u1 = np.array([np.cos(angle1), np.sin(angle1)])
            u2 = np.array([np.cos(angle2), np.sin(angle2)])

            r_diff = r2 - r1
            norm_r_diff = np.linalg.norm(r_diff) / L_ref

            if norm_r_diff == 0:
                continue  # éviter division par zéro

            term1 = np.dot(u2, u1) / norm_r_diff**3
            term2 = 3 * np.dot(u1, r_diff) * np.dot(u2, r_diff) / norm_r_diff**5

            energy += term1 - term2

    # On multiplie par les nombres des spinners pour tenir compte de leur "polarité"
    return energy * spinner1.number * spinner2.number


def total_energy(gate) -> float:
    """
    Calcule l'énergie totale pour tous les spinners du gate,
    somme sur toutes les paires distinctes.
    """
    spinners = [getattr(gate, f.name) for f in fields(gate) if isinstance(getattr(gate, f.name), Spinner)]
    E = 0.0
    for i in range(len(spinners)):
        for j in range(i + 1, len(spinners)):
            E += U(spinners[i], spinners[j])
    return E

def plot_spinner(spinner: Spinner, ax=None, branch_color='black', radius=0.005, branch_length=0.01):
    if ax is None:
        fig, ax = plt.subplots()

    x0, y0 = spinner.x, spinner.y
    theta = spinner.theta
    
    # Couleurs selon number (cercle)
    if spinner.number == 1:
        facecolor = 'black'     # cercle plein noir
        edgecolor = 'black'
    else:
        facecolor = 'white'     # cercle plein blanc
        edgecolor = 'black'

    # 3 branches épaisses espacées de 120°
    angles = [theta + i * 2 * np.pi / 3 for i in range(3)]
    for angle in angles:
        x_end = x0 + branch_length * np.cos(angle)
        y_end = y0 + branch_length * np.sin(angle)
        ax.plot([x0, x_end], [y0, y_end], color=branch_color, linewidth=10, zorder=1)

    # Cercle plein (au-dessus)
    circle = plt.Circle((x0, y0), radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=1, zorder=2)
    ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.autoscale(enable=True)

    # Suppression de la grille
    ax.grid(False)
    # Suppression des cadres (ticks, axes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plot_and_gate(gate):
    fig, ax = plt.subplots()

    # Couleurs selon nom de variable
    color_map = {
        'A': 'blue',
        'B': 'green',
        'C': 'orange',
        'F': 'purple',  # par exemple, si tu veux
    }

    for attr_name in dir(gate):
        attr = getattr(gate, attr_name)
        if isinstance(attr, Spinner):
            branch_color = color_map.get(attr_name, 'black')  # noir par défaut
            plot_spinner(attr, ax, branch_color=branch_color)

    plt.show()

def simulated_annealing(gate, T_init=1, T_min=0.0001, alpha=0.5, max_iter=1000):
    """
    Recuit simulé sur les angles theta des spinners dans la gate.
    T_init : température initiale
    T_min : température finale (arrêt)
    alpha : facteur de refroidissement
    max_iter : nombre d'itérations par température
    """
  
    current_gate = copy.deepcopy(gate)
    old_energy = total_energy(current_gate)
   
    T = T_init
    
    spinners = [getattr(current_gate, f.name) for f in fields(current_gate) if isinstance(getattr(current_gate, f.name), Spinner)]

    while T > T_min:
        for _ in range(max_iter):
            # Choisir un spinner au hasard
            spinner = random.choice(spinners)

            # Stocker l'ancien angle
            old_theta = spinner.theta

            # Modifier l'angle (petit pas aléatoire, ex: +/- 10 degrés)
            delta_theta = random.uniform(-10, 10)
            spinner.theta = (spinner.theta + delta_theta / 180 * np.pi) % (2 * np.pi)

            # Calculer nouvelle énergie
            new_energy = total_energy(current_gate)

            # Différence d'énergie
            delta_E = new_energy - old_energy

            # Critère d'acceptation
            if delta_E > 0:
                if random.random() > np.exp(-delta_E / T):
                    spinner.theta = old_theta
                    old_energy = new_energy

            else:
                old_energy = new_energy
        
        # Réduire la température
        T *= alpha
        print(f"T={T:.4f}, Energy={old_energy:.6f}")

    return current_gate, total_energy(current_gate)

# Exemple d'utilisation
and_gate = NOT()

print("Avant recuit:")
print(f"Energie initiale : {total_energy(and_gate)}")

optimized_gate, energy_min = simulated_annealing(and_gate)

print("Après recuit:")
print(f"Energie minimale : {energy_min}")

plot_and_gate(optimized_gate)

plot_and_gate(and_gate)
