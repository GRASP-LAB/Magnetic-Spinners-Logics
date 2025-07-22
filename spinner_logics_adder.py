from dataclasses import dataclass, field, fields
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import copy
import os

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

# Helper function to find nearest neighbors based on distance
def find_nearest_neighbors(gate_instance, distance_threshold: float = 1.2 * L) -> list[tuple[str, str]]:
    """
    Automatically identifies nearest neighbor pairs within a gate based on Euclidean distance.
    
    Args:
        gate_instance: An instance of a gate class (e.g., XOR, OR, etc.).
        distance_threshold: The maximum distance between spinner centers for them to be considered neighbors.
                            Default is 1.2 * L, which should cover direct connections.
                            Adjust this based on your gate's layout.
    
    Returns:
        A list of tuples, where each tuple contains the names of two neighboring spinners.
    """
    spinners_map = {f.name: getattr(gate_instance, f.name) 
                    for f in fields(gate_instance) 
                    if f.type == Spinner}
    
    neighbor_pairs = set() # Use a set to avoid duplicate pairs (e.g., (A,B) and (B,A))

    spinner_names = list(spinners_map.keys())
    num_spinners = len(spinner_names)

    for i in range(num_spinners):
        for j in range(i + 1, num_spinners):
            s1_name = spinner_names[i]
            s2_name = spinner_names[j]

            spinner1 = spinners_map[s1_name]
            spinner2 = spinners_map[s2_name]

            # Calculate Euclidean distance between spinner centers
            dist = np.sqrt((spinner1.x - spinner2.x)**2 + (spinner1.y - spinner2.y)**2)

            if dist <= distance_threshold:
                # Add the pair, ensuring consistent order (lexicographical)
                pair = tuple(sorted((s1_name, s2_name)))
                neighbor_pairs.add(pair)
                
    return sorted(list(neighbor_pairs)) # Convert back to list and sort for consistent output

lxorx = 0*Lx
lxory = -8*Ly

lxorx2 = 6*Lx
lxory2 = -9*Ly

lxorx3 = -5*Lx
lxory3 = -9*Ly

@dataclass
class adder:
    
    #xor 
    CC: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=0.0, y=6*Ly))
    B1: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=Lx, y=3*Ly))
    A1: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=3*Lx, y=Ly))
    C: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=5*Lx, y=Ly))
    F: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=2*Lx, y=6*Ly))

    C1: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx, y=0))
    C2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx, y=2*Ly))
    C3: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx, y=4*Ly))
    C4: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=3*Lx, y=5*Ly))
    C5: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=Lx, y=5*Ly))

    #cable

    C9: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=2*Lx, y=0*Ly))

    #xor
    A: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=0.0 + lxorx, y=6*Ly + lxory))
    B: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=Lx + lxorx, y=3*Ly + lxory))
    AA: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=3*Lx + lxorx, y=Ly + lxory))
    BB: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=5*Lx + lxorx, y=Ly + lxory))
    F1: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=0, x=2*Lx + lxorx, y=6*Ly + lxory))

    C11: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=4*Lx + lxorx, y=0 + lxory))
    C21: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=4*Lx + lxorx, y=2*Ly + lxory))
    C31: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=4*Lx + lxorx, y=4*Ly + lxory))
    C41: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=3*Lx + lxorx, y=5*Ly + lxory))
    C51: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=Lx + lxorx, y=5*Ly + lxory))

    #xor
    A3: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=0.0 + lxorx3, y=6*Ly + lxory3))
    B3: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=Lx + lxorx3, y=3*Ly + lxory3))
    AA3: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=3*Lx + lxorx3, y=Ly + lxory3))
    BB3: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=5*Lx + lxorx3, y=Ly + lxory3))
    F3: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=2*Lx + lxorx3, y=6*Ly + lxory3))

    C13: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx + lxorx3, y=0 + lxory3))
    C23: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx + lxorx3, y=2*Ly + lxory3))
    C33: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx + lxorx3, y=4*Ly + lxory3))
    C43: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=3*Lx + lxorx3, y=5*Ly + lxory3))
    C53: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=Lx + lxorx3, y=5*Ly + lxory3))
    #cable

    C91: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=2*Lx + lxorx3, y=8*Ly + lxory3))
    C92: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=2*Lx + lxorx3, y=10*Ly + lxory3))
    C93: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=3*Lx + lxorx3, y=11*Ly + lxory3))
    C94: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=4*Lx + lxorx3, y=12*Ly + lxory3))
    C95: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=5*Lx + lxorx3, y=12*Ly + lxory3))


    #xor
    A2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=0.0 + lxorx2, y=6*Ly + lxory2))
    B2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=Lx + lxorx2, y=3*Ly + lxory2))
    AA2: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=3*Lx + lxorx2, y=Ly + lxory2))
    BB2: Spinner = field(default_factory=lambda: Spinner(number=-1, theta=np.pi/2, x=5*Lx + lxorx2, y=Ly + lxory2))
    F2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=2*Lx + lxorx2, y=6*Ly + lxory2))

    C12: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx + lxorx2, y=0 + lxory2))
    C22: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx + lxorx2, y=2*Ly + lxory2))
    C32: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=4*Lx + lxorx2, y=4*Ly + lxory2))
    C42: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=3*Lx + lxorx2, y=5*Ly + lxory2))
    C52: Spinner = field(default_factory=lambda: Spinner(number=1, theta=np.pi/2, x=Lx + lxorx2, y=5*Ly + lxory2))
    
    #AND
    CC1: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=2*Lx + lxorx2, y=8*Ly + lxory2))
    CC2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=3*Lx + lxorx2, y=9*Ly + lxory2))
    CCC: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=4*Lx + lxorx2, y=8*Ly + lxory2))

    #CABLE
    CCC1: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=3*Lx + lxorx2, y=11*Ly + lxory2))
    CCC2: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=4*Lx + lxorx2, y=12*Ly + lxory2))
    CCC3: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=5*Lx + lxorx2, y=13*Ly + lxory2))
    CCC5: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=6*Lx + lxorx2, y=12*Ly + lxory2))
    CCC4: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=7*Lx + lxorx2, y=11*Ly + lxory2))


    FF: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=8*Lx + lxorx2, y=12*Ly + lxory2))

    #AND
    AAA: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=6*Lx + lxorx2, y=8*Ly + lxory2))
    CC3: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=7*Lx + lxorx2, y=9*Ly + lxory2))
    BBB: Spinner = field(default_factory=lambda: Spinner(number=1, theta=0, x=8*Lx + lxorx2, y=8*Ly + lxory2))


    N: int = 0
    neighbor_pairs: list[tuple[str, str]] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.N = sum(1 for f in fields(self) if f.type == Spinner)
        # Use a distance threshold to identify neighbors
        # You'll need to experiment with this value for complex gates like XOR.
        # A value like 1.5 * L (or even 2.0 * L) might work well for most direct connections.
        self.neighbor_pairs = find_nearest_neighbors(self, distance_threshold=1.1 * L) # Adjust as needed
        self._randomize_optimizable_thetas() # New: Randomize initial angles

    def _randomize_optimizable_thetas(self):
        # Define spinners that are 'inputs' and should retain their initial fixed theta.
        # All other spinners will have their theta randomized.
        fixed_spinners = {'A', 'B', 'C', 'D'} # Assuming A, B, C, D are inputs for the XOR gate
        for f in fields(self):
            if f.type == Spinner and f.name not in fixed_spinners:
                spinner = getattr(self, f.name)
                spinner.theta = random.uniform(0, 2 * np.pi) # Random angle between 0 and 360 degrees (0 to 2*pi radians)



def U(spinner1: Spinner, spinner2: Spinner) -> float:
    energy = 0
    L_ref = L - 2 * (R + l/2)
    
    theta1 = spinner1.theta
    theta2 = spinner2.theta

    x1, y1 = spinner1.x, spinner1.y
    x2, y2 = spinner2.x, spinner2.y

    for i in range(3):
        for j in range(3):
            angle1 = theta1 + i * 2 * np.pi / 3
            angle2 = theta2 + j * 2 * np.pi / 3

            r1 = np.array([
                x1 + (R + l/2) * np.cos(angle1),
                y1 + (R + l/2) * np.sin(angle1)
            ]) / L_ref

            r2 = np.array([
                x2 + (R + l/2) * np.cos(angle2),
                y2 + (R + l/2) * np.sin(angle2)
            ]) / L_ref

            u1 = np.array([np.cos(angle1), np.sin(angle1)])
            u2 = np.array([np.cos(angle2), np.sin(angle2)])

            r_diff = r2 - r1
            norm_r_diff = np.linalg.norm(r_diff) 

            if norm_r_diff == 0:
                continue

            term1 = np.dot(u2, u1) / norm_r_diff**3
            term2 = 3 * np.dot(u1, r_diff) * np.dot(u2, r_diff) / norm_r_diff**5

            energy += term1 - term2

    return energy * spinner1.number * spinner2.number


def total_energy_nearest_neighbors(gate) -> float:
    E = 0.0
    for s1_name, s2_name in gate.neighbor_pairs:
        spinner1 = getattr(gate, s1_name)
        spinner2 = getattr(gate, s2_name)
        E += U(spinner1, spinner2)
    return E

def plot_spinner(spinner: Spinner, ax=None, branch_color='black', radius=0.005, branch_length=0.008):
    if ax is None:
        fig, ax = plt.subplots()

    x0, y0 = spinner.x, spinner.y
    theta = spinner.theta
    
    if spinner.number == 1:
        facecolor = 'black'
        edgecolor = 'black'
    else:
        facecolor = 'white'
        edgecolor = 'black'

    angles = [theta + i * 2 * np.pi / 3 for i in range(3)]
    for angle in angles:
        x_end = x0 + branch_length * np.cos(angle)
        y_end = y0 + branch_length * np.sin(angle)
        ax.plot([x0, x_end], [y0, y_end], color=branch_color, linewidth=5, zorder=1)

    circle = plt.Circle((x0, y0), radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=1, zorder=2)
    ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.autoscale(enable=True)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plot_gate(gate, title="Gate Configuration"):
    fig, ax = plt.subplots(figsize=(10, 8))

    color_map = {
        'A': 'blue','AA': 'blue','AAA': 'blue','A2': 'blue','A3': 'blue','AA2': 'blue','AA3': 'blue', 'B': 'green', 'BB': 'green','BBB': 'green','B2': 'green','B3': 'green', 'BB2': 'green','BB3': 'green', 'C': 'pink','CC': 'pink','CCC': 'pink', 'D': 'green', 'F': 'purple','FF': 'purple'
    }

    for attr_name in dir(gate):
        attr = getattr(gate, attr_name)
        if isinstance(attr, Spinner):
            branch_color = color_map.get(attr_name, 'black')
            plot_spinner(attr, ax, branch_color=branch_color)

    if hasattr(gate, 'neighbor_pairs'):
        for s1_name, s2_name in gate.neighbor_pairs:
            spinner1 = getattr(gate, s1_name)
            spinner2 = getattr(gate, s2_name)
            ax.plot([spinner1.x, spinner2.x], [spinner1.y, spinner2.y], 'k--', linewidth=0.5, alpha=0.7)

    ax.set_title(title)
    plt.show()

def simulated_annealing(gate, T_init=1, T_min=0.0001, alpha=0.9, max_iter=500):
    current_gate = copy.deepcopy(gate)
    old_energy = total_energy_nearest_neighbors(current_gate)
   
    T = T_init
    
    optimizable_spinners = [
        getattr(current_gate, f.name)
        for f in fields(current_gate)
        if isinstance(getattr(current_gate, f.name), Spinner) and f.name not in {'A', 'B', 'C', 'D'}
    ]

    print(f"Starting simulated annealing for {type(gate).__name__}...")
    while T > T_min:
        for _ in range(max_iter):
            if not optimizable_spinners:
                break
            spinner = random.choice(optimizable_spinners)

            old_theta = spinner.theta

            delta_theta = 2 * random.choice([-1, 1]) 
            spinner.theta = (spinner.theta + np.radians(delta_theta)) % (2 * np.pi)

            new_energy = total_energy_nearest_neighbors(current_gate)

            delta_E = new_energy - old_energy

            if delta_E < 0:
                old_energy = new_energy
            else:
                if random.random() < np.exp(-delta_E / T):
                    old_energy = new_energy
                else:
                    spinner.theta = old_theta
        
        T *= alpha
        print(f"T={T:.4f}, Energy={old_energy:.6f}")

    return current_gate, total_energy_nearest_neighbors(current_gate), current_gate.F.theta

# Example Usage:
gate_to_simulate = adder()

print(f"--- Simulating {type(gate_to_simulate).__name__} Gate ---")
print(f"Initial Energy (nearest neighbors): {total_energy_nearest_neighbors(gate_to_simulate):.6f}")


# Plot initial configuration
plot_gate(gate_to_simulate, title=f"{type(gate_to_simulate).__name__} Gate - Initial Configuration")

#g, _, _ = simulated_annealing(gate_to_simulate, T_init=10, T_min=0.01, alpha=0.5, max_iter=100 * gate_to_simulate.N)

#plot_gate(g, title=f"{type(gate_to_simulate).__name__} Gate - Initial Configuration")


