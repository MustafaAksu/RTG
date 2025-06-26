import numpy as np
import pytest

from src.spherical_lattice import (
    generate_radial_shells,
    generate_fib_shell,
    build_spherical_lattice,
)


def test_generate_radial_shells_spacing():
    radii = generate_radial_shells(1.0, 4)
    assert len(radii) == 4
    expected = np.array([0.25, 0.5, 0.75, 1.0])
    assert np.allclose(radii, expected)


def test_generate_fib_shell_radius():
    points = generate_fib_shell(100, radius=2.0)
    # check that all points lie on sphere radius 2
    dists = np.linalg.norm(points, axis=1)
    assert np.allclose(dists, 2.0)


def test_build_spherical_lattice_shell_distances():
    lattice = build_spherical_lattice(1.0, 3, 10)
    radii = np.linalg.norm(lattice.reshape(3, 10, 3), axis=2)[:,0]
    expected = generate_radial_shells(1.0, 3)
    assert np.allclose(radii, expected)
