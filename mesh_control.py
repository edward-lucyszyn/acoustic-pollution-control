# -*- coding: utf-8 -*-

# Python packages
import scipy.sparse
import numpy as np
import matplotlib.pyplot

# Files import
from plot_functions import *
from extra import *

def build_node2elems(p_elem2nodes, elem2nodes):
    """Build numpy arrays nodes2elem and p_nodes2elem.""" 
    # elem2nodes connectivity matrix
    e2n_coef = np.ones(len(elem2nodes), dtype=np.int64)
    e2n_mtx = scipy.sparse.csr_matrix((e2n_coef, elem2nodes, p_elem2nodes))
    # node2elems connectivity matrix
    n2e_mtx = e2n_mtx.transpose()
    n2e_mtx = n2e_mtx.tocsr()
    # output
    p_node2elems = n2e_mtx.indptr
    node2elems = n2e_mtx.indices
    return p_node2elems, node2elems


def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid2nodes):
    """Add one element to the mesh. elemid2nodes must be a numpy array"""
    elem2nodes = np.append(elem2nodes, elemid2nodes)
    p_elem2nodes = np.append(p_elem2nodes, len(elem2nodes))
    return node_coords, p_elem2nodes, elem2nodes,


def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid_coords):
    """Add one node to the mesh. 
    The coords must be given is this format: np.array([[x, y, z]])."""
    node_coords = np.append(node_coords, nodeid_coords, axis=0)
    return node_coords, p_elem2nodes, elem2nodes,


def remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid):
    """Remove one element to the mesh."""
    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)

    # collecting the index of the points which only belong to element with id = elemid
    nodes_to_remove = np.array([], dtype=np.int64)
    for i in range(len(p_node2elems) - 1):
        if len(node2elems[p_node2elems[i]: p_node2elems[i + 1]]) == 1 and node2elems[p_node2elems[i]: p_node2elems[i + 1]][0] == elemid:
            nodes_to_remove = np.append(nodes_to_remove, i)
    
    diff = p_elem2nodes[elemid + 1] - p_elem2nodes[elemid]

    # delete points of element of index "elemid"
    elem2nodes = np.append(elem2nodes[:p_elem2nodes[elemid]], elem2nodes[p_elem2nodes[elemid + 1]:])

    # changing p_elem2nodes
    i = elemid + 1
    while i < len(p_elem2nodes):
        p_elem2nodes[i] -= diff
        i += 1

    p_elem2nodes = np.append(p_elem2nodes[:elemid], p_elem2nodes[elemid + 1:])
    
    # creating the new node_coords without the orphan points
    nodes_to_remove = list(set(nodes_to_remove))
    m = 0
    node_coords2 = np.array([])
    index_minus = np.array([])
    for i in range(len(node_coords)):
        if m < len(nodes_to_remove) and i == nodes_to_remove[m]:
            m += 1
        else:
            if len(node_coords2) == 0:
                node_coords2 = np.array([node_coords[i]])
            else:
                node_coords2 = np.append(node_coords2, np.array([node_coords[i]]), axis=0)
        index_minus = np.append(index_minus, np.array([m]))

    # changing elem2nodes with the index of the new node_coords
    for i in range(len(elem2nodes)):
        elem2nodes[i] -= index_minus[elem2nodes[i]]

    return node_coords2, p_elem2nodes, elem2nodes


def remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid):
    """Remove one node to mesh. It deletes all the elements also all the elements associated to the node."""

    # computing the id of all the elements that have the node to remove
    elem_to_remove = np.array([], dtype=np.int64)
    for i in range(len(p_elem2nodes) - 1):
        if nodeid in elem2nodes[p_elem2nodes[i]: p_elem2nodes[i + 1]]:
            elem_to_remove = np.append(elem_to_remove, i)
    
    # first case (that is supposed not to happen): there is no element to remove
    if len(elem_to_remove) == 0:

        # creating the new node_coords without the node whose id is nodeid
        m = 0
        node_coords2 = np.array([])
        index_minus = np.array([])
        for i in range(len(node_coords)):
            if m == 0 and i == nodeid:
                m = 1
            else:
                if len(node_coords2) == 0:
                    node_coords2 = np.array([node_coords[i]])
                else:
                    node_coords2 = np.append(node_coords2, np.array([node_coords[i]]), axis=0)
            index_minus = np.append(index_minus, np.array([m]))

        # changing elem2nodes with the index of the new node_coords
        for i in range(len(elem2nodes)):
            elem2nodes[i] -= index_minus[elem2nodes[i]]

    # second case: the point belongs to one or several elements. We delete all of them
    # it will delete the node and all the others orphan points since the function remove_elem_to_mesh deletes the orphan points
    for elemid in elem_to_remove[::-1]: # it is important to go backwards because after the id of the elements decrease
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid)
    
    return node_coords, p_elem2nodes, elem2nodes


def remove_elem_to_mesh_without_removing_nodes(node_coords, p_elem2nodes, elem2nodes, elemid):
    """Remove one element to the mesh without changing node_coords."""    
    diff = p_elem2nodes[elemid + 1] - p_elem2nodes[elemid]

    # delete points of element of index "elemid"
    elem2nodes = np.append(elem2nodes[:p_elem2nodes[elemid]], elem2nodes[p_elem2nodes[elemid + 1]:])

    # changing p_elem2nodes
    i = elemid + 1
    while i < len(p_elem2nodes):
        p_elem2nodes[i] -= diff
        i += 1

    p_elem2nodes = np.append(p_elem2nodes[:elemid], p_elem2nodes[elemid + 1:])
    
    return node_coords, p_elem2nodes, elem2nodes


def compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes, elemid):
    nodes = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid+1]]
    return np.average(node_coords[nodes,:], axis=0)


def compute_barycenter_all_elements(node_coords, p_elem2nodes, elem2nodes):
    """Compute the barycenter of one element."""
    spacedim = node_coords.shape[1]
    nelems = p_elem2nodes.shape[0] - 1
    elem_coords = np.zeros((nelems, spacedim), dtype=np.float64)
    for i in range(0, nelems):
        nodes = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        elem_coords[i,:] = np.average(node_coords[nodes,:], axis=0)
    return elem_coords


def create_regular_quadrangle_grid():
    """Generate grid with quadrangles."""
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 10, 10
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    node_coords = np.empty((nnodes, 2))
    borderid = np.array([], dtype=np.int64)
    for i in range(len(node_coords)):
        i_x = i%(nelemsx + 1)
        i_y = i//(nelemsx + 1)
        if i_x == 0 or i_x == nelemsx or i_y == 0 or i_y == nelemsy:
            borderid = np.append(borderid, i)
        node_coords[i] = [xmin + i_x * (xmax - xmin)/(nelemsx), ymin + i_y * (ymax - ymin)/(nelemsy)]
    
    p_elem2nodes = np.array([0], dtype=np.int64)
    elem2nodes = np.array([], dtype=np.int64)
    for i_y in range(nelemsy):
        for i_x in range(nelemsx):
            elem2nodes = np.append(elem2nodes, np.array([i_y * (nelemsx + 1) + i_x, i_y * (nelemsx + 1) + i_x + 1, (i_y + 1) * (nelemsx + 1) + i_x + 1, (i_y + 1) * (nelemsx + 1) + i_x]))
            p_elem2nodes = np.append(p_elem2nodes, len(elem2nodes))
    
    return node_coords, p_elem2nodes, elem2nodes, borderid


def shift_internal_nodes(node_coords, p_elem2nodes, elem2nodes, borderid):
    """Shift all the internal nodes slightly when knowing the id of the points at the border."""
    all_length = np.array([], dtype=np.float64)
    for i in range(len(p_elem2nodes) - 1):
        nodesid = elem2nodes[p_elem2nodes[i]: p_elem2nodes[i + 1]]
        all_length = np.append(all_length, np.linalg.norm(node_coords[nodesid[[0]]] - node_coords[nodesid[len(nodesid) - 1]], ord=2))
        for j in range(len(nodesid) - 1):
            all_length = np.append(all_length, np.linalg.norm(node_coords[nodesid[[j+1]]] - node_coords[nodesid[j]], ord=2))
    maximum_shift = np.min(all_length) /(2*np.sqrt(2))

    if 0 in borderid:
        node_coords2 = np.array([node_coords[0]])
    else:
        node_coords2 = np.array([[node_coords[0, 0] + np.random.uniform(-maximum_shift, maximum_shift), node_coords[0, 1] + np.random.uniform(-maximum_shift, maximum_shift)]])

    for i in range(1, len(node_coords)):
        if i in borderid:
            node_coords2 = np.append(node_coords2, np.array([[node_coords[i, 0], node_coords[i, 1]]]), axis=0)
        else:
            node_coords2 = np.append(node_coords2, np.array([[node_coords[i, 0] + np.random.uniform(-maximum_shift, maximum_shift), node_coords[i, 1] + np.random.uniform(-maximum_shift, maximum_shift)]]), axis=0)
    return node_coords2, p_elem2nodes, elem2nodes


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

    # Test for adding a node:
    print('\n Test for adding a node')
    print(node_coords[-1])
    node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([[1, 2, 0]]))
    print(node_coords[-1])

    # Test for adding an element:
    print('\n Test for adding an element')
    print(p_elem2nodes, '\n', elem2nodes)
    node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([2, 5, 6, 7]))
    print(p_elem2nodes, '\n', elem2nodes)

    # Test for removing a node:
    print('\n Test for removing a node')
    print(node_coords, p_elem2nodes, elem2nodes)
    node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 0)
    print(node_coords, p_elem2nodes, elem2nodes)

    # Test for removing an element:
    print('\n Test for removing an element')
    print(node_coords, p_elem2nodes, elem2nodes)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, 2)
    print(node_coords, p_elem2nodes, elem2nodes)

    # Test for creating the grid with quadrangles:
    node_coords, p_elem2nodes, elem2nodes, borderid = create_regular_quadrangle_grid()
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for creating the grid with quandrangles')
    matplotlib.pyplot.show()

    # Test for shifting internal nodes:
    node_coords, p_elem2nodes, elem2nodes = shift_internal_nodes(node_coords, p_elem2nodes, elem2nodes, borderid)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for shifting internal nodes')
    matplotlib.pyplot.show()
    
    pass


# Useless for now

def build_all_edges(node_coords, p_elem2nodes, elem2nodes):
    """Returns a list of the all the edges of a mesh."""
    all_edges = np.array([])
    for i in range(len(p_elem2nodes) - 1):
        nodesid = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i + 1]]
        for j in range(len(nodesid)):
            if len(all_edges) == 0:
                all_edges = np.array([order_normal_list_with_numpy_numbers([nodesid[j], nodesid[j+1]])])
            else:
                if j == len(nodesid) - 1:
                    all_edges = np.append(all_edges, np.array([order_normal_list_with_numpy_numbers([nodesid[j], nodesid[0]])]), axis=0)
                else:
                    all_edges = np.append(all_edges, np.array([order_normal_list_with_numpy_numbers([nodesid[j], nodesid[j+1]])]), axis=0)
    return all_edges
            
def _test_build_all_edges():
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
    print('\n Test for building all edges')
    print(build_all_edges(node_coords, p_elem2nodes, elem2nodes))