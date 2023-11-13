# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy as np
import scipy

# Files import
from mesh_control import *
from plot_functions import *
from extra import *
from fractal import *

# MRG packages
import zsolutions4students as solutions


def plot_errh(h_start, n, k=np.pi):
    """Plots error in different norms for values of h with a constant value of k (k=1 by default)."""
    L1 = np.array([], dtype=np.float64)
    L2 = np.array([], dtype=np.float64)
    Linf = np.array([], dtype=np.float64)
    H = np.array([], dtype=np.float64)

    # Reducing the size of the mesh by a factor 2 each time
    for i in range(n):
        H = np.append(H, h_start/(2**i))

    for h in H:
        xmin, xmax, ymin, ymax = 0, 1, 0, 1
        wavenumber = k
        nelemsx, nelemsy = int((xmax - xmin)/h), int((xmax - xmin)/h)
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1
        nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = np.unique(np.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )

        ###computing solexact
        solexacth = np.zeros((nnodes, 1), dtype=np.complex128)
        laplacian_of_solexact = np.zeros((nnodes, 1), dtype=np.complex128)
        for i in range(nnodes):
            x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
            solexacth[i] = np.exp(complex(0.,1.)*wavenumber*x)
            laplacian_of_solexact[i] = complex(0.,1.)*wavenumber*complex(0.,1.)*wavenumber * solexacth[i]

    # -- set dirichlet boundary conditions
        values_at_nodes_on_boundary = np.zeros((nnodes, 1), dtype=np.complex128)
        values_at_nodes_on_boundary[nodes_on_boundary] = solexacth[nodes_on_boundary]


    # -- set finite element matrices and right hand side
        f_unassembled = np.zeros((nnodes, 1), dtype=np.complex128)

        for i in range(nnodes):
            f_unassembled[i] = - laplacian_of_solexact[i] - (wavenumber ** 2) * solexacth[i]

        coef_k = np.ones((nelems, 1), dtype=np.complex128)#k = 1
        coef_m = np.ones((nelems, 1), dtype=np.complex128)
        K, M, F = solutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - wavenumber**2 * M
        B = F

    # -- apply Dirichlet boundary conditions
        A, B = solutions._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
        sol = scipy.linalg.solve(A, B)
        
    # -- compute error
        solrealh = sol.reshape((sol.shape[0], ))
        solexacth = solexacth.flatten()
        solrealh = solrealh.flatten()
        solerrh = solexacth - solrealh

        L1 = np.append(L1, np.linalg.norm(solerrh, 1))
        L2 = np.append(L2, np.linalg.norm(solerrh, 2))
        Linf = np.append(Linf, np.linalg.norm(solerrh, np.infty))

    matplotlib.pyplot.title('Error on the solution as a function of $h$')    
    matplotlib.pyplot.xlabel('$h$ in logarithmic scale')
    matplotlib.pyplot.ylabel('Error in logarithmic scale')
    reg = scipy.stats.linregress(np.log(H), np.log(L2))
    matplotlib.pyplot.plot(np.log(H), reg[0]*np.log(H) + reg[1], 'o-', label='Linear regression, $\\alpha$='+str(reg[0])[:4]+', $R^2$='+str(reg[2]))
    matplotlib.pyplot.plot(np.log(H), np.log(L1), 'g-', label='$L_1$')
    matplotlib.pyplot.plot(np.log(H), np.log(L2), 'b-', label='$L_2$')
    matplotlib.pyplot.plot(np.log(H), np.log(Linf), 'r-', label='$L_\infty$')
    matplotlib.pyplot.legend(title='Norms:')
    matplotlib.pyplot.show()


def plot_errk(k_min, k_max, n, h=2**(-4)):
    """Plots error in different norms for values of h with a constant value of k (k=1 by default)."""
    L1 = np.array([], dtype=np.float64)
    L2 = np.array([], dtype=np.float64)
    Linf = np.array([], dtype=np.float64)

    K_wave = np.linspace(np.log(k_min), np.log(k_max), n)
    K_wave = np.exp(K_wave)

    for k in K_wave:
        xmin, xmax, ymin, ymax = 0, 1, 0, 1
        wavenumber = k
        nelemsx, nelemsy = int((xmax - xmin)/h), int((xmax - xmin)/h)
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1
        nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = np.unique(np.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )

        ###computing solexact
        solexacth = np.zeros((nnodes, 1), dtype=np.complex128)
        laplacian_of_solexact = np.zeros((nnodes, 1), dtype=np.complex128)
        for i in range(nnodes):
            x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
            solexacth[i] = np.exp(complex(0.,1.)*wavenumber*x)
            laplacian_of_solexact[i] = complex(0.,1.)*wavenumber*complex(0.,1.)*wavenumber * solexacth[i]

        # -- set dirichlet boundary conditions
        values_at_nodes_on_boundary = np.zeros((nnodes, 1), dtype=np.complex128)
        values_at_nodes_on_boundary[nodes_on_boundary] = solexacth[nodes_on_boundary]


        # -- set finite element matrices and right hand side
        f_unassembled = np.zeros((nnodes, 1), dtype=np.complex128)

        for i in range(nnodes):
            f_unassembled[i] = - laplacian_of_solexact[i] - (wavenumber ** 2) * solexacth[i]

        coef_k = np.ones((nelems, 1), dtype=np.complex128)#k = 1
        coef_m = np.ones((nelems, 1), dtype=np.complex128)
        K, M, F = solutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - wavenumber**2 * M
        B = F

    # -- apply Dirichlet boundary conditions
        A, B = solutions._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
        sol = scipy.linalg.solve(A, B)
        
    # -- compute error
        solrealh = sol.reshape((sol.shape[0], ))
        solexacth = solexacth.flatten()
        solrealh = solrealh.flatten()
        solerrh = solexacth - solrealh

        L1 = np.append(L1, np.linalg.norm(solerrh, 1))
        L2 = np.append(L2, np.linalg.norm(solerrh, 2))
        Linf = np.append(Linf, np.linalg.norm(solerrh, np.infty))

    matplotlib.pyplot.title('Error on the solution as a function of $k$')    
    matplotlib.pyplot.xlabel('$k$ in logarithmic scale')
    matplotlib.pyplot.ylabel('Error in logarithmic scale')
    reg = scipy.stats.linregress(np.log(K_wave), np.log(L2))
    matplotlib.pyplot.plot(np.log(K_wave), reg[0]*np.log(K_wave) + reg[1], 'o-', label='Linear regression, $\\beta$='+str(reg[0])[:4]+', $R^2$='+str(reg[2]))
    matplotlib.pyplot.plot(np.log(K_wave), np.log(L1), 'g-', label='$L_1$')
    matplotlib.pyplot.plot(np.log(K_wave), np.log(L2), 'b-', label='$L_2$')
    matplotlib.pyplot.plot(np.log(K_wave), np.log(Linf), 'r-', label='$L_\infty$')
    matplotlib.pyplot.legend(title='Norms:')
    matplotlib.pyplot.show()


def plot_errh_with_shift(h_start, n, k=np.pi):
    """Plots error in different norms for values of h with a constant value of k (k=1 by default)."""
    L1 = np.array([], dtype=np.float64)
    L2 = np.array([], dtype=np.float64)
    Linf = np.array([], dtype=np.float64)
    H = np.array([], dtype=np.float64)

    # Reducing the size of the mesh by a factor 2 each time
    for i in range(n):
        H = np.append(H, h_start/(2**i))

    for h in H:
        xmin, xmax, ymin, ymax = 0, 1, 0, 1
        wavenumber = k
        nelemsx, nelemsy = int((xmax - xmin)/h), int((xmax - xmin)/h)
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1
        nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = np.unique(np.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )

        ###computing solexact
        solexacth = np.zeros((nnodes, 1), dtype=np.complex128)
        laplacian_of_solexact = np.zeros((nnodes, 1), dtype=np.complex128)
        for i in range(nnodes):
            x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
            solexacth[i] = np.exp(complex(0.,1.)*wavenumber*x)
            laplacian_of_solexact[i] = complex(0.,1.)*wavenumber*complex(0.,1.)*wavenumber * solexacth[i]

    # -- set dirichlet boundary conditions
        values_at_nodes_on_boundary = np.zeros((nnodes, 1), dtype=np.complex128)
        values_at_nodes_on_boundary[nodes_on_boundary] = solexacth[nodes_on_boundary]


    # -- set finite element matrices and right hand side
        f_unassembled = np.zeros((nnodes, 1), dtype=np.complex128)

        for i in range(nnodes):
            f_unassembled[i] = - laplacian_of_solexact[i] - (wavenumber ** 2) * solexacth[i]

        coef_k = np.ones((nelems, 1), dtype=np.complex128)#k = 1
        coef_m = np.ones((nelems, 1), dtype=np.complex128)
        K, M, F = solutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - wavenumber**2 * M
        B = F

    # -- apply Dirichlet boundary conditions
        A, B = solutions._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
        sol = scipy.linalg.solve(A, B)
        
    # -- compute error
        solrealh = sol.reshape((sol.shape[0], ))
        solexacth = solexacth.flatten()
        solrealh = solrealh.flatten()
        solerrh = solexacth - solrealh

        L1 = np.append(L1, np.linalg.norm(solerrh, 1))
        L2 = np.append(L2, np.linalg.norm(solerrh, 2))
        Linf = np.append(Linf, np.linalg.norm(solerrh, np.infty))

    matplotlib.pyplot.title('Error on the solution as a function of $h$')    
    matplotlib.pyplot.xlabel('$h$ in logarithmic scale')
    matplotlib.pyplot.ylabel('Error in logarithmic scale')
    reg = scipy.stats.linregress(np.log(H), np.log(L2))
    matplotlib.pyplot.plot(np.log(H), reg[0]*np.log(H) + reg[1], 'o-', label='Linear regression, $\\alpha$='+str(reg[0])[:4]+', $R^2$='+str(reg[2]))
    matplotlib.pyplot.plot(np.log(H), np.log(L1), 'g-', label='$L_1$')
    matplotlib.pyplot.plot(np.log(H), np.log(L2), 'b-', label='$L_2$')
    matplotlib.pyplot.plot(np.log(H), np.log(Linf), 'r-', label='$L_\infty$')
    matplotlib.pyplot.legend(title='Norms:')
    matplotlib.pyplot.show()


if __name__ == '__main__':
    # plot_errh(2**(-2), 5, np.pi)
    f_min, f_max = 20, 200
    c = 340 # speed of sound
    k_min, k_max = 2*np.pi*f_min/c, 2*np.pi*f_max/c
    plot_errk(k_min, k_max, 50, 0.05)
    pass

