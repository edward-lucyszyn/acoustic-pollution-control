# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy as np

# Files import
from delaunay_triangulation import *
from plot_functions import *
from mesh_control import *


def swap_diagonals(node_coords, p_elem2nodes, elem2nodes, elemid1, elemid2):
    """Swap the diagonal between two connected triangles."""
    elem1 = elem2nodes[p_elem2nodes[elemid1]: p_elem2nodes[elemid1 + 1]]
    elem2 = elem2nodes[p_elem2nodes[elemid2]: p_elem2nodes[elemid2 + 1]]
    actual_diagonal = np.array([], dtype=np.int64)
    new_diagonal = np.array([], dtype=np.int64)

    for node in elem1:
        if node in elem2:
            actual_diagonal = np.append(actual_diagonal, node)
        else:
            new_diagonal = np.append(new_diagonal, node)
    for node in elem2:
        if node not in elem1:
            new_diagonal = np.append(new_diagonal, node)

    if is_counterclockwise(node_coords, np.array([new_diagonal[0], new_diagonal[1], actual_diagonal[0]])):
        elem2nodes[p_elem2nodes[elemid1]] = new_diagonal[0]
        elem2nodes[p_elem2nodes[elemid1] + 1] = new_diagonal[1]
        elem2nodes[p_elem2nodes[elemid1] + 2] = actual_diagonal[0]
    else:
        elem2nodes[p_elem2nodes[elemid1]] = new_diagonal[1]
        elem2nodes[p_elem2nodes[elemid1] + 1] = new_diagonal[0]
        elem2nodes[p_elem2nodes[elemid1] + 2] = actual_diagonal[0]

    if is_counterclockwise(node_coords, np.array([new_diagonal[0], new_diagonal[1], actual_diagonal[1]])):
        elem2nodes[p_elem2nodes[elemid2]] = new_diagonal[0]
        elem2nodes[p_elem2nodes[elemid2] + 1] = new_diagonal[1]
        elem2nodes[p_elem2nodes[elemid2] + 2] = actual_diagonal[1]
    else:
        elem2nodes[p_elem2nodes[elemid2]] = new_diagonal[1]
        elem2nodes[p_elem2nodes[elemid2] + 1] = new_diagonal[0]
        elem2nodes[p_elem2nodes[elemid2] + 2] = actual_diagonal[1]

    return node_coords, p_elem2nodes, elem2nodes

def _test_swap_diagonals():
    node_coords = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
        ])
    elem2nodes = np.array([0, 1, 2, 0, 2, 3])
    p_elem2nodes = np.array([0, 3, 6])
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for swap_diagonals')
    matplotlib.pyplot.show()
    node_coords, p_elem2nodes, elem2nodes = swap_diagonals(node_coords, p_elem2nodes, elem2nodes, 0, 1)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for swap_diagonals')
    matplotlib.pyplot.show()


def is_quadrilateral_convex(node_coords, n0, n1, n2, n3):
    """Returns True if a quadrilateral is stricly convex (is convex and doesn't have 3 points aligned)."""
    A = np.cross(node_coords[n2] - node_coords[n1], node_coords[n1] - node_coords[n0])
    B = np.cross(node_coords[n3] - node_coords[n2], node_coords[n2] - node_coords[n1])
    C = np.cross(node_coords[n0] - node_coords[n3], node_coords[n3] - node_coords[n2])
    D = np.cross(node_coords[n1] - node_coords[n0], node_coords[n0] - node_coords[n3])
    return np.sign(A) == np.sign(B) == np.sign(C) == np.sign(D)

def _test_is_quadrilateral_convex():
    node_coords = np.array([
        [0.6, 0.6],
        [0, 1],
        [1, 1],
        [1, 0]
        ])
    print('\n Test for is_quadrilateral_convex:')
    print(is_quadrilateral_convex(node_coords, 0, 1, 2, 3))


def add_constraint_to_triangulation(node_coords, p_elem2nodes, elem2nodes, e1, e2):
    """Adds one constraint to the triangulation. It there is a point on the segment of the 
    constraint, the algorithm does as if there are two edges, from e1 to the point aligned 
    and fom the point aligned to e2."""
    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)
    reached = False
    first_aligned_point = len(node_coords)
    intersecting_edges = np.array([], dtype=np.int64)
    intersecting_triangles = np.array([], dtype=np.int64)
    elem_id_added = np.array([], dtype=np.int64)

    j = 0 # looking for a triangle with e1
    # for all trianles with e1, either the triangle has also e2, either the triangle doesn't have e2
    # in this last case: there is exactly one triangle that countains e1 in the mesh with one side that intersects with e1e2

    while j < len(node2elems[p_node2elems[e1]: p_node2elems[e1 + 1]]):
        elemid = node2elems[p_node2elems[e1]: p_node2elems[e1 + 1]][j]
        elem = elem2nodes[p_elem2nodes[elemid]: p_elem2nodes[elemid + 1]]
        if e2 in elem: # the constraint is respected, no need to do anything
            j = len(node2elems[p_node2elems[e1]: p_node2elems[e1 + 1]])
            reached = True
        elif do_intersect(node_coords[elem[0]], node_coords[elem[1]], node_coords[e1], node_coords[e2]): # we look for an intersection
            intersecting_edges = np.array([elem[0], elem[1]], dtype=np.int64)
            intersecting_triangles = np.array([elemid], dtype=np.int64)
            j = len(node2elems[p_node2elems[e1]: p_node2elems[e1 + 1]])
        elif do_intersect(node_coords[elem[1]], node_coords[elem[2]], node_coords[e1], node_coords[e2]):
            intersecting_edges = np.array([elem[1], elem[2]], dtype=np.int64)
            intersecting_triangles = np.array([elemid], dtype=np.int64)
            j = len(node2elems[p_node2elems[e1]: p_node2elems[e1 + 1]])
        elif do_intersect(node_coords[elem[2]], node_coords[elem[0]], node_coords[e1], node_coords[e2]):
            intersecting_edges = np.array([elem[2], elem[0]], dtype=np.int64)
            intersecting_triangles = np.array([elemid], dtype=np.int64)
            j = len(node2elems[p_node2elems[e1]: p_node2elems[e1 + 1]])
        elif elem[0] != e1 and is_in_segment(node_coords[e1], node_coords[e2], node_coords[elem[0]]): # case where the node is on the constraint
            return add_constraint_to_triangulation(node_coords, p_elem2nodes, elem2nodes, elem[0], e2)
        elif elem[1] != e1 and is_in_segment(node_coords[e1], node_coords[e2], node_coords[elem[1]]):
            return add_constraint_to_triangulation(node_coords, p_elem2nodes, elem2nodes, elem[1], e2)
        elif elem[2] != e1 and is_in_segment(node_coords[e1], node_coords[e2], node_coords[elem[2]]):
            return add_constraint_to_triangulation(node_coords, p_elem2nodes, elem2nodes, elem[2], e2)
        else:
            j += 1

    if len(intersecting_edges) == 0 and not reached:
        raise AssertionError("Troubles with finding the right triangle to start with.", node2elems[p_node2elems[e1]:p_node2elems[e1+1]], node2elems[p_node2elems[e2]:p_node2elems[e2+1]])
    
    while not reached:
        common_elem = np.intersect1d(node2elems[p_node2elems[intersecting_edges[-1]]: p_node2elems[intersecting_edges[-1] + 1]], node2elems[p_node2elems[intersecting_edges[-2]]: p_node2elems[intersecting_edges[-2] + 1]])
        if common_elem[0] == elemid:
            elemid = common_elem[1]
        else:
            elemid = common_elem[0]

        elem = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
        if e2 in elem: # the constraint is respected, no need to do anything
            intersecting_triangles = np.append(intersecting_triangles, elemid)
            reached = True
        elif do_intersect(node_coords[elem[0]], node_coords[elem[1]], node_coords[e1], node_coords[e2]) and (np.sort(np.array([elem[0], elem[1]])) == np.sort(intersecting_edges[-2:])).all() == False:
            intersecting_edges = np.append(intersecting_edges, [elem[0], elem[1]])
            intersecting_triangles = np.append(intersecting_triangles, elemid)
        elif do_intersect(node_coords[elem[1]], node_coords[elem[2]], node_coords[e1], node_coords[e2]) and (np.sort(np.array([elem[1], elem[2]])) == np.sort(intersecting_edges[-2:])).all() == False:
            intersecting_edges = np.append(intersecting_edges, [elem[1], elem[2]])
            intersecting_triangles = np.append(intersecting_triangles, elemid)
        elif do_intersect(node_coords[elem[2]], node_coords[elem[0]], node_coords[e1], node_coords[e2]) and (np.sort(np.array([elem[2], elem[0]])) == np.sort(intersecting_edges[-2:])).all() == False:
            intersecting_edges = np.append(intersecting_edges, [elem[2], elem[0]])
            intersecting_triangles = np.append(intersecting_triangles, elemid)
        elif elem[0] != e1 and is_in_segment(node_coords[e1], node_coords[e2], node_coords[elem[0]]):
            first_aligned_point = elem[0]
            reached = True
            intersecting_triangles = np.append(intersecting_triangles, elemid)
        elif elem[1] != e1 and is_in_segment(node_coords[e1], node_coords[e2], node_coords[elem[1]]):
            first_aligned_point = elem[1]
            reached = True
            intersecting_triangles = np.append(intersecting_triangles, elemid)
        elif elem[2] != e1 and is_in_segment(node_coords[e1], node_coords[e2], node_coords[elem[2]]):
            first_aligned_point = elem[2]
            reached = True
            intersecting_triangles = np.append(intersecting_triangles, elemid)
        else:
            raise AssertionError("No intersection or e2 found")

    left_side = np.array([e1], dtype=np.int64)
    right_side = np.array([e1], dtype=np.int64)
    for nodeid in intersecting_edges:
            if is_counterclockwise(node_coords, np.array([e1, e2, nodeid])) and nodeid not in left_side :
                left_side = np.append(left_side, nodeid)
            elif not is_counterclockwise(node_coords, np.array([e1, e2, nodeid])) and nodeid not in right_side:
                right_side = np.append(right_side, nodeid)
    
    if first_aligned_point < len(node_coords):
        left_side = np.append(left_side, first_aligned_point)
        right_side = np.append(right_side, first_aligned_point)        
    else:
        left_side = np.append(left_side, e2)
        right_side = np.append(right_side, e2)

    p_elem2nodes_left, elem2nodes_left = apply_non_convex_Delaunay_triangulation(node_coords, np.array([left_side]))[1:]
    p_elem2nodes_right, elem2nodes_right = apply_non_convex_Delaunay_triangulation(node_coords, np.array([right_side]))[1:]

    for elemid in np.sort(intersecting_triangles)[::-1]:
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh_without_removing_nodes(node_coords, p_elem2nodes, elem2nodes, elemid)
    
    for i in range(len(p_elem2nodes_left) - 1):
        node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elem2nodes_left[p_elem2nodes_left[i]: p_elem2nodes_left[i + 1]])

    for i in range(len(p_elem2nodes_right) - 1):
        node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elem2nodes_right[p_elem2nodes_right[i]: p_elem2nodes_right[i + 1]])

    if first_aligned_point < len(node_coords):
        node_coords, p_elem2nodes, elem2nodes = add_constraint_to_triangulation(node_coords, p_elem2nodes, elem2nodes, first_aligned_point, e2)

    return node_coords, p_elem2nodes, elem2nodes


def apply_constrained_Delaunay_triangulation(point_coords, nodesid, constraints):
    """Apply a constrained Delaunay triangulation to the points in point_coords 
    whose indexes are in nodesid. constraints is an array of forced edges, it means 
    that constraints is to the form np.array([[id1, id2], [id1, id2]])."""
    node_coords, p_elem2nodes, elem2nodes = apply_Delaunay_triangulation(point_coords, nodesid)

    i = 0
    while i < len(constraints): # for every constraints
        node_coords, p_elem2nodes, elem2nodes = add_constraint_to_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints[i][0], constraints[i][1])
        i += 1
        
    return node_coords, p_elem2nodes, elem2nodes

def _test_constrained_Delaunay_triangulation():
    point_coords = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [1.5, 0],
        [1.5, 1],
        [1.5, 2],
    ])
    c1, c2 = 0, 5

    nodesid = np.array([i for i in range(len(point_coords))])
    node_coords, p_elem2nodes, elem2nodes = apply_Delaunay_triangulation(point_coords, nodesid)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.plot(np.array([node_coords[c1][0], node_coords[c2][0]]), np.array([node_coords[c1][1], node_coords[c2][1]]), color='red')
    matplotlib.pyplot.title('Test for applying constrained Delaunay algorithm')
    matplotlib.pyplot.show()

    node_coords, p_elem2nodes, elem2nodes = apply_constrained_Delaunay_triangulation(point_coords, nodesid, np.array([[c1, c2]]))
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for applying constrained Delaunay algorithm')
    matplotlib.pyplot.show()

def _test_constrained_Delaunay_triangulation2():
    point_coords = generate_random_point_uniformly(10, 20)
    nodesid = np.array([i for i in range(len(point_coords))])
    node_coords, p_elem2nodes, elem2nodes = apply_Delaunay_triangulation(point_coords, nodesid)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    c1, c2 = 0, 5
    matplotlib.pyplot.title('Test for applying constrained Delaunay algorithm')
    matplotlib.pyplot.plot(np.array([node_coords[c1][0], node_coords[c2][0]]), np.array([node_coords[c1][1], node_coords[c2][1]]), color='red')
    matplotlib.pyplot.show()
    
    node_coords, p_elem2nodes, elem2nodes = apply_constrained_Delaunay_triangulation(point_coords, nodesid, np.array([[c1, c2]]))
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for applying constrained Delaunay algorithm')
    matplotlib.pyplot.show()


def apply_constrained_non_convex_Delaunay_triangulation(point_coords, constraints, borders):
    """Apply a constrained Delaunay triangulation with a non convex form to the points 
    in point_coords whose indexes are in nodesid. constraints is an array of forced edges, 
    it means that constraints is to the form np.array([[id1, id2], [id1, id2]])."""
    nodesid = np.array([], np.int64)
    for i in range(len(borders)):
        for j in range(len(borders[i])):
            nodesid = np.append(nodesid, borders[i][j])
    node_coords, p_elem2nodes, elem2nodes = apply_constrained_Delaunay_triangulation(point_coords, nodesid, constraints)
    elem_to_delete = np.array([], dtype=np.int64)
    for elemid in range(len(p_elem2nodes) - 1):
        barycenter_coords = compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes, elemid)
        if not is_inside_form(node_coords, borders, barycenter_coords) and elemid not in elem_to_delete:
            elem_to_delete = np.append(elem_to_delete, elemid)
    for elemid in np.sort(elem_to_delete)[::-1]:
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid)

    return node_coords, p_elem2nodes, elem2nodes

def _test_constrained_non_convex_Delaunay_triangulation():
    c1, c2 = 6, 23
    c3, c4 = 8, 16
    motif = [[0.25, 0], [0.25, 0.25], [0.5, 0.25], [0.5, 0], [0.5, -0.25], [0.75, -0.25], [0.75, 0]]
    point_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    border = np.array([0, 1, 2, 3, 0])
    point_coords, border = generate_fractal_border(point_coords, border, motif, 1)
    border = border[:-1]
    borders = np.array([border])
    node_coords, p_elem2nodes, elem2nodes = apply_non_convex_Delaunay_triangulation(point_coords, borders)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.plot(np.array([node_coords[c1][0], node_coords[c2][0]]), np.array([node_coords[c1][1], node_coords[c2][1]]), color='red')
    matplotlib.pyplot.plot(np.array([node_coords[c3][0], node_coords[c4][0]]), np.array([node_coords[c3][1], node_coords[c4][1]]), color='red')
    matplotlib.pyplot.title('Test for applying constrained non convex Delaunay triangulation')
    matplotlib.pyplot.show()

    constraints = np.array([[c1, c2], [c3, c4]])
    node_coords, p_elem2nodes, elem2nodes = apply_constrained_non_convex_Delaunay_triangulation(point_coords, constraints, borders)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for applying constrained non convex Delaunay triangulation')
    matplotlib.pyplot.show()

def _test_constrained_non_convex_Delaunay_triangulation2():
    c1, c2 = 1, 3
    motif = [[0.25, 0], [0.25, 0.25], [0.5, 0.25], [0.5, 0], [0.5, -0.25], [0.75, -0.25], [0.75, 0]]
    point_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    border = np.array([0, 1, 2, 3, 0])
    point_coords, border = generate_fractal_border(point_coords, border, motif, 2)
    border = border[:-1]
    borders = np.array([border])
    node_coords, p_elem2nodes, elem2nodes = apply_non_convex_Delaunay_triangulation(point_coords, borders)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.plot(np.array([node_coords[c1][0], node_coords[c2][0]]), np.array([node_coords[c1][1], node_coords[c2][1]]), color='red')
    matplotlib.pyplot.title('Test for applying constrained non convex Delaunay triangulation')
    matplotlib.pyplot.show()

    constraints = np.array([[c1, c2]])
    node_coords, p_elem2nodes, elem2nodes = apply_constrained_non_convex_Delaunay_triangulation(point_coords, constraints, borders)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for applying constrained non convex Delaunay triangulation')
    matplotlib.pyplot.show()

if __name__ == '__main__':
    # _test_swap_diagonals()
    # _test_is_quadrilateral_convex()
    _test_constrained_Delaunay_triangulation()
    _test_constrained_Delaunay_triangulation2()
    _test_constrained_non_convex_Delaunay_triangulation()
    _test_constrained_non_convex_Delaunay_triangulation2()
    pass


