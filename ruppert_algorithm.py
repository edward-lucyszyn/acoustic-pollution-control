# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy as np

# Files import
from plot_functions import *
from mesh_control import *
from delaunay_triangulation import *
from constrained_delaunay_triangulation import *


def is_a_bad_triangle(node_coords, p_elem2nodes, elem2nodes, elemid, max_area, min_angle):
    """Return True if the tringle of id elemid is a triangle with an area greater than 
    max_area of with an angle less than min_angle."""
    elem = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
    area = triangle_area(node_coords[elem[0]], node_coords[elem[1]], node_coords[elem[2]])
    min_angle_of_elem = np.min(triangle_angles_in_degrees(node_coords[elem[0]], node_coords[elem[1]], node_coords[elem[2]]))
    if min_angle_of_elem < min_angle or area > max_area:
        return True
    else:
        return False
    

def does_mesh_have_a_bad_angle(node_coords, p_elem2nodes, elem2nodes, min_angle):
    """Return True if there is an element of the mesh with at least one edge which has too small angle. 
    Return also the index of the element with the worst angle."""
    min_angle = np.infty
    min_index = len(p_elem2nodes)
    for i in range(len(p_elem2nodes) - 1):
        elem = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i + 1]]
        min_angle_of_elem = np.min(triangle_angles_in_degrees(node_coords[elem[0]], node_coords[elem[1]], node_coords[elem[2]]))
        if min_angle_of_elem < min_angle:
            min_angle = min_angle_of_elem
            min_index = i
    return min_angle < min_angle_of_elem, min_index


def does_mesh_have_a_bad_area(node_coords, p_elem2nodes, elem2nodes, max_area):
    """Return True if there is an element of the mesh with at least one edge which has too small angle. 
    Return also the index of the element with the worst angle."""
    area = 0
    max_index = len(p_elem2nodes)
    for i in range(len(p_elem2nodes) - 1):
        elem = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i + 1]]
        elem_area = triangle_area(node_coords[elem[0]], node_coords[elem[1]], node_coords[elem[2]])
        if area < elem_area:
            area = elem_area
            max_index = i
    return area > max_area, max_index
    

def is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, nodeid1, nodeid2, coords):
    """Return True if node of the point of coordinates coords is in the diametral circle of the segment formed by nodeid1, nodeid2."""
    return np.linalg.norm(((node_coords[nodeid1] + node_coords[nodeid2])/2) - coords) < np.linalg.norm((node_coords[nodeid1] - node_coords[nodeid2])/2)


def does_edge_have_a_node_in_its_diametral_circle(node_coords, p_elem2nodes, elem2nodes, nodeid1, nodeid2):
    """Return True if there is a node of the mesh in the diametral circle of the segment defined by nodeid1 and nodeid2."""
    i = 0
    while i < len(node_coords):
        if i != nodeid1 and i != nodeid2:
            if is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, nodeid1, nodeid2, node_coords[i]):
                return True
        i += 1
    return False


def does_element_have_a_wrong_edge(node_coords, p_elem2nodes, elem2nodes, elemid):
    """Return True if there is an element of the mesh with at least one edge which has a node in its diametral circle. 
    Return also the indexes of the nodes of the segment whose diametral circle contains nodes."""
    elem = elem2nodes[p_elem2nodes[elemid]: p_elem2nodes[elemid + 1]]
    for i in range(len(elem)):
        if does_edge_have_a_node_in_its_diametral_circle(node_coords, p_elem2nodes, elem2nodes, elem[i], elem[(i+1)%len(elem)]):
            return True, elem[i], elem[(i+1)%len(elem)]
    return False, len(node_coords), len(node_coords)


def _test_is_node_in_diametral_circle():
    node_coords = np.array([[0, 0], [2, 0]])
    coords1 = np.array([1, 1])
    coords2 = np.array([1, 0.5])
    coords3 = np.array([1, 2])
    coords4 = np.array([2, 0])
    p_elem2nodes, elem2nodes = np.array([0]), np.array([])
    assert is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, 0, 1, coords1) == False, "Error in is_node_in_diametral_circle"
    assert is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, 0, 1, coords2) == True, "Error in is_node_in_diametral_circle"
    assert is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, 0, 1, coords3) == False, "Error in is_node_in_diametral_circle"
    assert is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, 0, 1, coords4) == False, "Error in is_node_in_diametral_circle"


def is_stranger_node_in_a_diametral_circle(node_coords, p_elem2nodes, elem2nodes, coords):
    """Return True or False as first argument and returns the indexes of the segment the nearest segment."""
    r_min = np.infty
    i_min1, i_min2 = len(node_coords), len(node_coords)
    for i in range(len(p_elem2nodes) - 1):
        elem = elem2nodes[p_elem2nodes[i]: p_elem2nodes[i + 1]]
        for j in range(len(elem)):
            if is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, elem[j], elem[(j+1)%len(elem)], coords) and np.linalg.norm(((node_coords[elem[j]] + node_coords[elem[(j+1)%len(elem)]])/2) - coords) < r_min:
                r_min = np.linalg.norm(((node_coords[elem[j]] + node_coords[elem[(j+1)%len(elem)]])/2) - coords)
                i_min1, i_min2 = elem[j], elem[(j+1)%len(elem)]
    return r_min != np.infty, i_min1, i_min2

def is_node_in_a_diametral_circle(node_coords, p_elem2nodes, elem2nodes, nodeid):
    """Return True or False as first argument and returns the indexes of the segment the nearest segment."""
    r_min = np.infty
    i_min1, i_min2 = len(node_coords), len(node_coords)
    for i in range(len(p_elem2nodes) - 1):
        elem = elem2nodes[p_elem2nodes[i]: p_elem2nodes[i + 1]]
        for j in range(len(elem)):
            if is_node_in_diametral_circle(node_coords, p_elem2nodes, elem2nodes, elem[j], elem[(j+1)%len(elem)], node_coords[nodeid]) and np.linalg.norm(((node_coords[elem[j]] + node_coords[elem[(j+1)%len(elem)]])/2) - node_coords[nodeid]) < r_min:
                r_min = np.linalg.norm(((node_coords[elem[j]] + node_coords[elem[(j+1)%len(elem)]])/2) - node_coords[nodeid])
                i_min1, i_min2 = elem[j], elem[(j+1)%len(elem)]
    return r_min != np.infty, i_min1, i_min2

def _test_is_stranger_node_in_a_diametral_circle():
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elem2nodes = np.array([0, 1, 2, 2, 3, 0])
    p_elem2nodes = np.array([0, 3, 6])
    coords = np.array([-3, 0.875])
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.plot(coords[0], coords[1], 'o', color='red')
    matplotlib.pyplot.title('Test for is_stranger_node_in_a_diametral_circle')
    matplotlib.pyplot.show()
    print('\n Test for is_edge_in_a_demetral circle:')
    print(is_stranger_node_in_a_diametral_circle(node_coords, p_elem2nodes, elem2nodes, coords))


def add_node_to_non_convex_constrained_delaunay_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints, borders, coords):
    """Add one point to a non convex constrained Delaunay triangulation. The point coordinates are in the array coords."""
    if not is_inside_form(node_coords, borders, coords):
        return node_coords, p_elem2nodes, elem2nodes, np.array([])
    else:
        node_coords, p_elem2nodes, elem2nodes = add_one_point_to_triangulation(node_coords, p_elem2nodes, elem2nodes, np.array([coords]), 0)
        i = 0
        while i < len(constraints): # for every constraints
            node_coords, p_elem2nodes, elem2nodes= add_constraint_to_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints[i][0], constraints[i][1])
            i += 1
    return node_coords, p_elem2nodes, elem2nodes

def _test_add_node_to_non_convex_constrained_delaunay_triangulation():
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elem2nodes = np.array([0, 1, 2, 2, 3, 0])
    p_elem2nodes = np.array([0, 3, 6])
    coords1 = np.array([-3, 0.875])
    coords2 = np.array([0.4, 0.6])
    coords3 = np.array([0.6, 0.4])
    constraints = np.array([[0, 2]])
    borders = np.array([[0, 1, 2, 3]])

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for adding a point to a non convex constrained Delaunay triangulation')
    matplotlib.pyplot.show()

    node_coords, p_elem2nodes, elem2nodes = add_node_to_non_convex_constrained_delaunay_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints, borders, coords1)
    node_coords, p_elem2nodes, elem2nodes = add_node_to_non_convex_constrained_delaunay_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints, borders, coords2)
    node_coords, p_elem2nodes, elem2nodes = add_node_to_non_convex_constrained_delaunay_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints, borders, coords3)

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for adding a point to a non convex constrained Delaunay triangulation')
    matplotlib.pyplot.show()


def _test_add_node_to_non_convex_constrained_delaunay_triangulation2():
    node_coords = np.array([[0, 0], [0.5, 0.5], [1, 0.5], [1.5, 0.5], [2, 0]])
    elem2nodes = np.array([2, 1, 0, 4, 3, 2, 4, 2, 0])
    p_elem2nodes = np.array([0, 3, 6, 9])
    coords = np.array([1, 0])
    constraints = np.array([])
    borders = np.array([[0, 1, 2, 3, 4]])

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for adding a point to a non convex constrained Delaunay triangulation')
    matplotlib.pyplot.show()

    node_coords, p_elem2nodes, elem2nodes, elem_id_deleted, elem_id_added = split_segment(node_coords, p_elem2nodes, elem2nodes, constraints, borders, 0, 4)

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for adding a point to a non convex constrained Delaunay triangulation')
    matplotlib.pyplot.show()


def split_segment(node_coords, p_elem2nodes, elem2nodes, constraints, borders, nodeid1, nodeid2):
    """Adds the midpoint of a segment into the triangulation. Continues until there is no node in diametral circles of this segment."""
    coords = (node_coords[nodeid1] + node_coords[nodeid2])/2
    node_coords, p_elem2nodes, elem2nodes = add_node_to_non_convex_constrained_delaunay_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints, borders, coords)
    new_index = len(node_coords) - 1
    result1 = does_edge_have_a_node_in_its_diametral_circle(node_coords, p_elem2nodes, elem2nodes, nodeid1, new_index)
    result2 = does_edge_have_a_node_in_its_diametral_circle(node_coords, p_elem2nodes, elem2nodes, nodeid2, new_index)
    if result1:
        node_coords, p_elem2nodes, elem2nodes = split_segment(node_coords, p_elem2nodes, elem2nodes, constraints, borders, nodeid1, new_index)
    if result2:
        node_coords, p_elem2nodes, elem2nodes = split_segment(node_coords, p_elem2nodes, elem2nodes, constraints, borders, nodeid2, new_index)
    return node_coords, p_elem2nodes, elem2nodes

def _test_split_segment():
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.4, 0.5]])
    elem2nodes = np.array([0, 1, 2, 2, 3, 0])
    p_elem2nodes = np.array([0, 3, 6])
    nodeid1, nodeid2 = 0, 2
    constraints = np.array([[0, 2]])
    borders = np.array([[0, 1, 2, 3]])

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for split_segment')
    matplotlib.pyplot.show()

    node_coords, p_elem2nodes, elem2nodes = split_segment(node_coords, p_elem2nodes, elem2nodes, constraints, borders, nodeid1, nodeid2)

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for split_segment')
    matplotlib.pyplot.show()


def compute_circumcenter_coords(node_coords, p_elem2nodes, elem2nodes, elemid):
    """Returns the coords of the circumcenter."""
    elem = elem2nodes[p_elem2nodes[elemid]: p_elem2nodes[elemid + 1]]
    a, b, c = elem
    alpha, beta, gamma = triangle_angles(node_coords[a], node_coords[b], node_coords[c])
    x_c = (node_coords[a][0]*np.sin(2*alpha) + node_coords[b][0]*np.sin(2*beta) + node_coords[c][0]*np.sin(2*gamma))/(np.sin(2*alpha) + np.sin(2*beta) + np.sin(2*gamma))
    y_c = (node_coords[a][1]*np.sin(2*alpha) + node_coords[b][1]*np.sin(2*beta) + node_coords[c][1]*np.sin(2*gamma))/(np.sin(2*alpha) + np.sin(2*beta) + np.sin(2*gamma))
    return x_c, y_c


def add_node_at_circumcenter(node_coords, p_elem2nodes, elem2nodes, constraints, borders, elemid):
    """Adds a node at the circumcenter of the triangle of id elemid. Then computes 
    the Delaunay triangulation with this new point with respect to the constraints."""
    x_c, y_c = compute_circumcenter_coords(node_coords, p_elem2nodes, elem2nodes, elemid)
    coords = np.array([x_c, y_c])
    result , nodeid1, nodeid2 = is_stranger_node_in_a_diametral_circle(node_coords, p_elem2nodes, elem2nodes, coords)
    if result:
        return split_segment(node_coords, p_elem2nodes, elem2nodes, constraints, borders, nodeid1, nodeid2)
    else:
        return add_node_to_non_convex_constrained_delaunay_triangulation(node_coords, p_elem2nodes, elem2nodes, constraints, borders, coords)
    
def _test_add_node_at_circumcenter():
    node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    elem2nodes = np.array([0, 1, 2, 2, 3, 0])
    p_elem2nodes = np.array([0, 3, 6])
    elemid = 1
    constraints = np.array([[0, 2]])
    borders = np.array([[0, 1, 2, 3]])

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for add_node_at_circumcenter')
    matplotlib.pyplot.show()

    node_coords, p_elem2nodes, elem2nodes = add_node_at_circumcenter(node_coords, p_elem2nodes, elem2nodes, constraints, borders, elemid)

    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_node(node_coords, p_elem2nodes, elem2nodes)
    matplotlib.pyplot.title('Test for add_node_at_circumcenter')
    matplotlib.pyplot.show()


def compute_elem_changes(old_p_elem2nodes, old_elem2nodes, new_p_elem2nodes, new_elem2nodes):
    """Returns the numpy array of the indexes of the element deleted and the index of the new elements."""
    m = 0
    added_elem = np.array([], dtype=np.int64)
    old_to_new = np.array([], dtype=np.int64)
    for i in range(len(old_p_elem2nodes) - 1):
        if m < len(new_p_elem2nodes) - 1 and (old_elem2nodes[old_p_elem2nodes[i]: old_p_elem2nodes[i+1]] == new_elem2nodes[new_p_elem2nodes[m]: new_p_elem2nodes[m+1]]).all() == True:
            old_to_new = np.append(old_to_new, m)
            m += 1
        else:
            old_to_new = np.append(old_to_new, -1)
    while m < len(new_p_elem2nodes) - 1:
        added_elem = np.append(added_elem, m)
        m += 1
    return added_elem, old_to_new


def apply_Ruppert_algorithm(point_coords, constraints, borders, max_area, min_angle, max_iter=100):
    """Apply the Ruppert algorithm with counstraints and borders. Every triangle will have 
    an area under max_area and angles over min_angle. min_angle is in degree."""
    node_coords, p_elem2nodes, elem2nodes = apply_constrained_non_convex_Delaunay_triangulation(point_coords, constraints, borders)
    elem_to_replace = np.array([], dtype=np.int64)
    n = 0

    for i in range(len(p_elem2nodes) - 1):
        if is_a_bad_triangle(node_coords, p_elem2nodes, elem2nodes, i, max_area, min_angle) or does_element_have_a_wrong_edge(node_coords, p_elem2nodes, elem2nodes, i)[-1]:
           elem_to_replace = np.append(elem_to_replace, i)
    
    while n < max_iter and len(elem_to_replace) > 0:
        changed = False
        elemid = elem_to_replace[0]
        elem = elem2nodes[p_elem2nodes[elemid]: p_elem2nodes[elemid + 1]]
        old_p_elem2nodes, old_elem2nodes = p_elem2nodes, elem2nodes
        for i in range(3):
            nodeid1, nodeid2 = elem[i], elem[(i+1)%3]
            if not changed and does_edge_have_a_node_in_its_diametral_circle(node_coords, p_elem2nodes, elem2nodes, nodeid1, nodeid2):
                node_coords, p_elem2nodes, elem2nodes = split_segment(node_coords, p_elem2nodes, elem2nodes, constraints, borders, nodeid1, nodeid2)
                changed = True

        if not changed and is_a_bad_triangle(node_coords, p_elem2nodes, elem2nodes, elemid, max_area, min_angle):
            node_coords, p_elem2nodes, elem2nodes = add_node_at_circumcenter(node_coords, p_elem2nodes, elem2nodes, constraints, borders, elemid)
            changed = True

        if changed:
            elem_added, old_to_new = compute_elem_changes(old_p_elem2nodes, old_elem2nodes, p_elem2nodes, elem2nodes)

            new_elem_to_replace = np.array([], dtype=np.int64)
            for i in range(len(elem_to_replace)):
                if old_to_new[elem_to_replace[i]] != -1:
                    new_elem_to_replace = np.append(new_elem_to_replace, old_to_new[elem_to_replace[i]])
                
            for id in elem_added:
                if is_a_bad_triangle(node_coords, p_elem2nodes, elem2nodes, id, max_area, min_angle) or does_element_have_a_wrong_edge(node_coords, p_elem2nodes, elem2nodes, id)[-1]:
                    elem_to_replace = np.append(elem_to_replace, id)

        else:
            elem_to_replace = elem_to_replace[1:]
        n += 1

    return node_coords, p_elem2nodes, elem2nodes


def _test_apply_Ruppert_algorithm():
    point_coords = np.array([
        [0, 0],
        [0, 2],
        [2, 2],
        [2, 0],
        [0.5, 1],
        [1, 1],
        [1, 0.5],
        [0.5, 0.5]])
    constraints = np.array([])
    borders = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    max_area = np.infty
    min_angle = 22

    node_coords, p_elem2nodes, elem2nodes = apply_constrained_non_convex_Delaunay_triangulation(point_coords, constraints, borders)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for Ruppert algorithm')
    matplotlib.pyplot.show()
    node_coords, p_elem2nodes, elem2nodes = apply_Ruppert_algorithm(point_coords, constraints, borders, max_area, min_angle)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for Ruppert algorithm')
    matplotlib.pyplot.show()


def _test_apply_Ruppert_algorithm2():
    motif = [[0.25, 0], [0.25, 0.25], [0.5, 0.25], [0.5, 0], [0.5, -0.25], [0.75, -0.25], [0.75, 0]]
    point_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    border = np.array([0, 1, 2, 3])
    point_coords, border = generate_fractal_border(point_coords, border, motif, 2)
    borders = np.array([border])
    constraints = np.array([])
    max_area = np.infty
    min_angle = 0

    node_coords, p_elem2nodes, elem2nodes = apply_constrained_non_convex_Delaunay_triangulation(point_coords, constraints, borders)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for Ruppert algorithm')
    matplotlib.pyplot.show()
    node_coords, p_elem2nodes, elem2nodes = apply_Ruppert_algorithm(point_coords, constraints, borders, max_area, min_angle)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for Ruppert algorithm')
    matplotlib.pyplot.show()

            
def test():
    node_coords = np.array([[0, 0], [0, 2], [2, 2], [2, 0], [0.5, 1], [1, 1], [1, 0.5], [0.5, 0.5], [0.25, 0.5], [0.5, 1.5], [1.5, 0.5], [1, 0]]) 
    p_elem2nodes = np.array([0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]) 
    elem2nodes = np.array([1,  8,  4,  0,  8,  1,  0,  7,  8 , 4,  8 , 7 , 1 , 4 , 9 , 4 , 5 , 9 , 2 , 9 , 5 , 1,  9,  2,
  2,  5, 10,  2, 10,  3,  5,  6, 10,  0, 11,  3,  6,  7, 11,  0, 11,  7,  3, 10, 11,  6, 11, 10])
    

if __name__ == "__main__":
    # _test_split_segment()
    # _test_add_node_at_circumcenter()
    # _test_is_node_in_diametral_circle()
    # _test_add_node_to_non_convex_constrained_delaunay_triangulation()
    # _test_add_node_to_non_convex_constrained_delaunay_triangulation2()
    # _test_split_segment()
    # _test_add_node_at_circumcenter()
    # _test_is_edge_in_a_diametral_circle()
    _test_apply_Ruppert_algorithm()
    # _test_apply_Ruppert_algorithm2()
    pass