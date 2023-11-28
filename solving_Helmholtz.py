# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy as np
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


# File given by the university
import zsolutions4students as solutions

# Files import
from plot_functions import *
from mesh_control import *
from fractal import *
from delaunay_triangulation import *


def solve_Helmholtz_equation(node_coords, p_elem2nodes, elem2nodes, nodes_on_boundary, wavenumber, values_at_nodes_on_boundary, f_unassembled, save=False, show=True):
    nelems = len(p_elem2nodes) - 1
    
    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
    matplotlib.pyplot.show()

    coef_k = np.ones((nelems, 1), dtype=np.complex128)
    coef_m = np.ones((nelems, 1), dtype=np.complex128)
    K, M, F = solutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M + complex(0.0, 0.1) * 10 * M
    # A = K - wavenumber**2 * M
    B = F

    A, B = solutions._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)
    sol = scipy.linalg.solve(A, B)

    solreal = sol.reshape((sol.shape[0], ))

    _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, np.real(solreal))
    _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, np.imag(solreal))

    # plot the eigenvalues
    eigenvals = scipy.linalg.eigvals(A)
    matplotlib.pyplot.scatter(np.real(eigenvals), np.imag(eigenvals))
    matplotlib.pyplot.show()


def _test_solve_Helmholtz_equation():
    motif = [[0.25, 0], [0.25, 0.25], [0.5, 0.25], [0.5, 0], [0.5, -0.25], [0.75, -0.25], [0.75, 0]]
    point_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    border = np.array([0, 1, 2, 3])
    point_coords, border = generate_fractal_border(point_coords, border, motif, 2)
    constraints = np.array([])
    # border = np.append(border, np.array([2, 3]))
    nodes_on_boundary = np.array([3, 0])
    node_coords, p_elem2nodes, elem2nodes = apply_non_convex_Delaunay_triangulation(point_coords, np.array([border]))
    values_at_nodes_on_boundary = np.zeros((node_coords.shape[0], 1), dtype=np.complex128)
    f_unassembled = np.ones((len(node_coords), 1), dtype=np.complex128)
    for wavenumber in [np.pi, 20*np.pi]:
        solve_Helmholtz_equation(node_coords, p_elem2nodes, elem2nodes, nodes_on_boundary, wavenumber, values_at_nodes_on_boundary, f_unassembled, save=False, show=True)


if __name__ == '__main__':
    _test_solve_Helmholtz_equation()
    pass


