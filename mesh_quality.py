# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy as np

# Files import
from plot_functions import *


def compute_aspect_ratio(node_coords, p_elem2nodes, elem2nodes, elemid):
    """Compute the aspect ratio of either a triangle or either a quadrangle."""
    nodesid = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid+1]]

    # Triangles case
    if len(nodesid) == 3:
        a = np.linalg.norm(node_coords[nodesid[1]] - node_coords[nodesid[0]], ord=2)
        b = np.linalg.norm(node_coords[nodesid[2]] - node_coords[nodesid[1]], ord=2)
        c = np.linalg.norm(node_coords[nodesid[0]] - node_coords[nodesid[2]], ord=2)
        s = (a+b+c)/2
        rho = np.sqrt(((s-a)*(s-b)*(s-c))/s)
        return (np.sqrt(3)/6)*np.max([a, b, c])/rho
    
    else: # Quadrangle case
        e_0 = node_coords[nodesid[1]] - node_coords[nodesid[0]]
        e_0 = e_0/np.linalg.norm(e_0, ord=2)
        e_1 = node_coords[nodesid[2]] - node_coords[nodesid[1]]
        e_1 = e_1/np.linalg.norm(e_1, ord=2)
        e_2 = node_coords[nodesid[3]] - node_coords[nodesid[2]]
        e_2 = e_2/np.linalg.norm(e_2, ord=2)
        e_3 = node_coords[nodesid[0]] - node_coords[nodesid[3]]
        e_3 = e_3/np.linalg.norm(e_3, ord=2)
        S = np.abs(np.dot(e_0, e_1)) + np.abs(np.dot(e_1, e_2)) + np.abs(np.dot(e_2, e_3)) + np.abs(np.dot(e_3, e_0))
        return 1 - S/4
    

def compute_edge_length_factor(node_coords, p_elem2nodes, elem2nodes, elemid):
    nodesid = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid+1]]
    all_lengths = np.array([np.linalg.norm(node_coords[nodesid[0]] - node_coords[nodesid[len(nodesid) - 1]], ord=2)], dtype=np.float64)
    for i in range(len(nodesid) - 1):
        all_lengths = np.append(all_lengths, np.linalg.norm(node_coords[nodesid[i + 1]] - node_coords[nodesid[i]], ord=2))
    return np.min(all_lengths)/np.mean(all_lengths)


def analysis_of_the_mesh(node_coords, p_elem2nodes, elem2nodes):
    """Save and show two histograms that give an analysis of mesh's quality."""
    if len(p_elem2nodes) == 1:
        return
    else:
        all_aspect_ratio = np.array([], dtype=np.float64)
        all_edge_length_factor = np.array([], dtype=np.float64)
        if len(elem2nodes[p_elem2nodes[0]:p_elem2nodes[1]]) == 3: # then the elements are triangles
            for i in range(len(p_elem2nodes) - 1):
                all_aspect_ratio = np.append(all_aspect_ratio, 1/(compute_aspect_ratio(node_coords, p_elem2nodes, elem2nodes, i)))
        else:
            for i in range(len(p_elem2nodes) - 1):
                all_aspect_ratio = np.append(all_aspect_ratio, compute_aspect_ratio(node_coords, p_elem2nodes, elem2nodes, i))
        for i in range(len(p_elem2nodes) - 1):
            all_edge_length_factor = np.append(all_edge_length_factor, compute_edge_length_factor(node_coords, p_elem2nodes, elem2nodes, i))
        aspect_ratio_histogram = np.histogram(all_aspect_ratio, 10, (0.0, 1.0))
        edge_length_factor_histogram = np.histogram(all_edge_length_factor, 10, (0.0, 1.0))

        matplotlib.pyplot.stairs(edge_length_factor_histogram[0], edge_length_factor_histogram[1], fill=True)
        matplotlib.pyplot.title('Histogram of edge length factors')
        matplotlib.pyplot.savefig('Edge length', dpi=1000)
        matplotlib.pyplot.show()

        matplotlib.pyplot.stairs(aspect_ratio_histogram[0], aspect_ratio_histogram[1], fill=True)
        matplotlib.pyplot.title('Histogram of aspect ratios')
        matplotlib.pyplot.savefig('Aspect ratios', dpi=1000)
        matplotlib.pyplot.show()
    
if __name__ == '__main__':
    # Creating node_coords, elem2nodes, p_elem2nodes for the tests
    node_coords = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [1.5, 0, 0],
        [1.5, 1, 0],
        [1.5, 2, 0],
    ])

    elem2nodes = np.array([0, 1, 2, 3, 1, 2, 5, 4, 2, 5, 6])

    p_elem2nodes = np.array([0, 4, 8, 11])

    # Test for computing aspect ratio:
    print('\n Test of aspect ratio')
    print(compute_aspect_ratio(node_coords, p_elem2nodes, elem2nodes, 0))
    print(compute_aspect_ratio(node_coords, p_elem2nodes, elem2nodes, 1))
    print(compute_aspect_ratio(node_coords, p_elem2nodes, elem2nodes, 2))

    # Test for computing edge length factor:
    print('\n Test of edge length factor')
    print(compute_edge_length_factor(node_coords, p_elem2nodes, elem2nodes, 0))
    print(compute_edge_length_factor(node_coords, p_elem2nodes, elem2nodes, 1))
    print(compute_edge_length_factor(node_coords, p_elem2nodes, elem2nodes, 2))

    # Test of the analysis of the mesh:
    analysis_of_the_mesh(node_coords, p_elem2nodes, elem2nodes)