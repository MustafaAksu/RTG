"""Simple visualization of a spherical lattice."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.spherical_lattice import build_spherical_lattice


def main():
    lattice = build_spherical_lattice(max_radius=1.0, n_shells=4, points_per_shell=100)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(lattice[:, 0], lattice[:, 1], lattice[:, 2], s=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    main()
