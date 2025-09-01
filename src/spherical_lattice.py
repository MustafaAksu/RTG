import numpy as np


def generate_radial_shells(max_radius, n_shells):
    """Return radii for `n_shells` evenly spaced from 0 to `max_radius`."""
    if n_shells <= 0:
        raise ValueError("n_shells must be positive")
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")
    return np.linspace(0, max_radius, n_shells + 1)[1:]


def generate_fib_shell(n_points, radius=1.0):
    """Generate points on a sphere using the Fibonacci sphere algorithm."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices / n_points)
    golden_ratio = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * indices / golden_ratio

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    return np.vstack((x, y, z)).T


def build_spherical_lattice(max_radius, n_shells, points_per_shell):
    """Create a spherical lattice of Fibonacci shells."""
    shells = []
    radii = generate_radial_shells(max_radius, n_shells)
    for r in radii:
        shells.append(generate_fib_shell(points_per_shell, r))
    return np.vstack(shells)
