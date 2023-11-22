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


def is_in_circumcenter(node_coords, triangles_id):
    """If trangles_id = numpy.array([a b c d]), a b c are in counterclockwise sens, 
    then return True if D is in the circumcircle of triangle abc."""
    M = np.array([
        [node_coords[triangles_id[0]][0] - node_coords[triangles_id[3]][0], node_coords[triangles_id[0]][1] - node_coords[triangles_id[3]][1], node_coords[triangles_id[0]][0]**2 - node_coords[triangles_id[3]][0]**2 + node_coords[triangles_id[0]][1]**2 - node_coords[triangles_id[3]][1]**2],
        [node_coords[triangles_id[1]][0] - node_coords[triangles_id[3]][0], node_coords[triangles_id[1]][1] - node_coords[triangles_id[3]][1], node_coords[triangles_id[1]][0]**2 - node_coords[triangles_id[3]][0]**2 + node_coords[triangles_id[1]][1]**2 - node_coords[triangles_id[3]][1]**2],
        [node_coords[triangles_id[2]][0] - node_coords[triangles_id[3]][0], node_coords[triangles_id[2]][1] - node_coords[triangles_id[3]][1], node_coords[triangles_id[2]][0]**2 - node_coords[triangles_id[3]][0]**2 + node_coords[triangles_id[2]][1]**2 - node_coords[triangles_id[3]][1]**2]
    ])
    return np.linalg.det(M) > 0

def _test_is_in_circumcenter():
    node_coords = np.array([[0, 0], [1, 1], [0, 2], [0, 1]])
    triangles_id = np.array([0, 1, 2, 3])
    print('\n Test for is_in_circumcenter')
    print(is_in_circumcenter(node_coords, triangles_id))


def generate_random_point_uniformly(size, number_of_points):
    """Generate n=number_of_points points in a square centered in 0 and of side's length
    equals to size with a uniform law."""
    return np.random.uniform(low=-size/2, high=size/2, size=(number_of_points, 2))

def _test_generate_random_point_uniformly():
    print('Test for generating random points uniformly')
    print(generate_random_point_uniformly(10, 19)[[1, 2, 3, 4],0])


def is_counterclockwise(node_coords, triangles_id):
    """With triangles_id = np.array([a, b, c]), returns True if the triangle is counterclockwise."""
    return (node_coords[triangles_id[1]][0] - node_coords[triangles_id[0]][0])*(node_coords[triangles_id[2]][1] - node_coords[triangles_id[0]][1]) - (node_coords[triangles_id[1]][1] - node_coords[triangles_id[0]][1])*(node_coords[triangles_id[2]][0] - node_coords[triangles_id[0]][0]) > 0

def _test_is_counterclockwise():
    node_coords = generate_random_point_uniformly(1, 3)
    trianglesid = [0, 1, 2]
    result = is_counterclockwise(node_coords, trianglesid)

    matplotlib.pyplot.plot(node_coords[0][0], node_coords[0][1], 'bo', color = 'red', label='A')
    matplotlib.pyplot.plot(node_coords[1][0], node_coords[1][1], 'bo', color='green', label='B')
    matplotlib.pyplot.plot(node_coords[2][0], node_coords[2][1], 'bo', color='blue', label='C')
    matplotlib.pyplot.title('Test for is_counterclockwise function. Here it returns ' + str(result))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


def construct_fictive_triangle(point_coords, pointsid):
    """Create a triangle around the nodes representend by pointsid."""
    x_min, x_max = np.min(point_coords[pointsid,0:1]), np.max(point_coords[pointsid,0:1])
    y_min, y_max = np.min(point_coords[pointsid,1:2]), np.max(point_coords[pointsid,1:2])
    A = np.array([[x_min - 5*(x_max - x_min), y_min - 10*(y_max - y_min)]])
    B = np.array([[x_max + 5*(x_max - x_min), y_min - 10*(y_max - y_min)]])
    C = np.array([[x_min + 0.5*(x_max - x_min), y_max + 10*(y_max - y_min)]])
    node_coords = np.array(A)
    p_elem2nodes = np.array([0], dtype=np.int64)
    elem2nodes = np.array([], dtype=np.int64)
    for node_array in [B, C]:
        node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, node_array)
    node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([len(node_coords) - 3, len(node_coords) - 2, len(node_coords) - 1], dtype=np.int64))
    return node_coords, p_elem2nodes, elem2nodes

def _test_construct_fictive_triangle():
    point_coords = generate_random_point_uniformly(10, 20)
    node_coords, p_elem2nodes, elem2nodes = construct_fictive_triangle(point_coords, np.array([i for i in range(len(point_coords))]))
    plot_elem(node_coords, p_elem2nodes, elem2nodes, 0, colorname='orange')
    plot_all_node(point_coords, p_elem2nodes, elem2nodes, colorname='blue')
    matplotlib.pyplot.title('Test for constructing fictive triange')
    matplotlib.pyplot.show()


def add_one_point_to_triangulation(node_coords, p_elem2nodes, elem2nodes, point_coords, pointid, show_changes=False):
    """Add one point to the Delaunay triangulation. The point coordinates are in the array point_coords 
    at the index pointid."""
    node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([point_coords[pointid]]))
    nodeid = len(node_coords) - 1
    elem_id_to_delete = np.array([], dtype=np.int64)
    elem_id_added = np.array([], dtype=np.int64)
    triangles_to_delete = []
    # find the triangles where point whose id is pointid is in the circumcircle
    for i in range(len(p_elem2nodes) - 1):
        nodes_id = np.append(elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]], nodeid)
        if is_in_circumcenter(node_coords, nodes_id):
            elem_id_to_delete = np.append(elem_id_to_delete, i)
            triangles_to_delete.append(elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]])

    # plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    # plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    # for elemid in elem_id_to_delete:
    #     plot_elem(node_coords, p_elem2nodes, elem2nodes, elemid, colorname='blue')
    # matplotlib.pyplot.plot(point_coords[pointid][0], point_coords[pointid][1], 'bo', color='blue')
    # matplotlib.pyplot.title('Test for finding')
    # matplotlib.pyplot.show(

    # then the all the points ot the triangles that we removed form a connex polygon whose sides need to be joined with the new point to construct triangles
    all_sides = []
    for triangle in triangles_to_delete:
        all_sides.append(order_normal_list_with_numpy_numbers([triangle[0], triangle[1]]))
        all_sides.append(order_normal_list_with_numpy_numbers([triangle[1], triangle[2]]))
        all_sides.append(order_normal_list_with_numpy_numbers([triangle[2], triangle[0]]))

    all_unique_sides = []
    for i in range(len(all_sides)):
        j = 0
        while j < len(all_sides):
            if j != i and all_sides[i] == all_sides[j]:
                j = len(all_sides) + 1
            j += 1
        if j == len(all_sides):
            all_unique_sides.append(all_sides[i])

    # if it is a unique side, then it makes a triangle with the new point
    for points in all_unique_sides:
        if not is_in_segment(node_coords[points[0]], node_coords[points[1]], node_coords[nodeid]):
            if is_counterclockwise(node_coords, np.array([points[0], points[1], nodeid])):
                node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([points[0], points[1], nodeid]))
            else:
                node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([points[0], nodeid, points[1]]))

            elem_id_added = np.append(elem_id_added, len(p_elem2nodes) - 1 - len(elem_id_to_delete))

    # removing the triangles to the mesh
    for elem_id in  np.sort(elem_id_to_delete)[::-1]:
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elem_id)

    if show_changes:
        return node_coords, p_elem2nodes, elem2nodes, elem_id_to_delete, elem_id_added
    else:
        return node_coords, p_elem2nodes, elem2nodes

def _test_add_one_point_to_triangulation():
    nbre = 20
    point_coords = generate_random_point_uniformly(10, nbre)
    plot_all_node(point_coords, np.array([0]), np.array([]))
    matplotlib.pyplot.title('Test for adding one point to the triangulation')
    matplotlib.pyplot.show()
    node_coords, p_elem2nodes, elem2nodes = construct_fictive_triangle(point_coords, np.array([i for i in range(len(point_coords))]))
    nbre  = 20
    for i in range(nbre):
        node_coords, p_elem2nodes, elem2nodes = add_one_point_to_triangulation(node_coords, p_elem2nodes, elem2nodes, point_coords, i)
        plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
        plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
        matplotlib.pyplot.title('Test for adding one point to the triangulation at step=' + str(i))
        matplotlib.pyplot.show()


def apply_Delaunay_triangulation(point_coords, nodesid):
    """Triangulate with Delaunay's triangulation the points in the numpy array nodesid.
    This should return a convex triangulation (should because sometimes if two nodes of 
    the boundary are too close the fictive triangle isn't big enough to considerate the 
    edge between these two points)."""
    node_coords, p_elem2nodes, elem2nodes = construct_fictive_triangle(point_coords, nodesid)
    nodesid = np.sort(nodesid)
    m = 0
    for i in range(len(point_coords)):
        if m < len(nodesid) and i == nodesid[m]:
            m += 1
            node_coords, p_elem2nodes, elem2nodes = add_one_point_to_triangulation(node_coords, p_elem2nodes, elem2nodes, point_coords, i)
        else:
            node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([point_coords[i]]))
    for _ in range(3):
        node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 0)
    return node_coords, p_elem2nodes, elem2nodes

def _test_apply_Delaunay_triangulation():
    point_coords = generate_random_point_uniformly(10, 50)
    nodesid = np.array([i for i in range(len(point_coords))])
    plot_all_node(point_coords, np.array([0]), np.array([]), colorname='red')
    matplotlib.pyplot.title('Test for applying Delaunay triangulation')
    matplotlib.pyplot.show()
    node_coords, p_elem2nodes, elem2nodes = apply_Delaunay_triangulation(point_coords, nodesid)
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange')
    plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red')
    matplotlib.pyplot.title('Test for applying Delaunay triangulation')
    matplotlib.pyplot.show()


# Delaunay triangulation with a defined border (not necessarily convex)

def do_intersect(a1, b1, a2, b2, eps=1e-10):
    """Returns True if the segment [a1, b1] intersects with [a2, b2]. a1 , a2, ... are numpy arrays."""
    U = np.dot(np.cross(np.array(b1 - a1), np.array(b2 - b1)), np.cross(np.array(b1 - a1), np.array(a2 - b1))) < -eps
    V = np.dot(np.cross(np.array(b2 - a2), np.array(b1 - b2)), np.cross(np.array(b2 - a2), np.array(a1 - b2))) < -eps
    return U and V

def _test_do_intersect():
    a1 = np.array([0.75, -1])
    a2 = np.array([0.75, 0.25])
    b2 = np.array([1, 0.25])
    b1 = np.array([0.5, -0.25])
    result = do_intersect(a1, a2, b1, b2)
    matplotlib.pyplot.plot([a1[0], a2[0]], [a1[1], a2[1]])
    matplotlib.pyplot.plot([b1[0], b2[0]], [b1[1], b2[1]])
    U, V = compute_intersect_values(a1, a2, b1, b2)
    matplotlib.pyplot.title('Test for do_intersect. Here it returns ' + str(result) + ' with U=' + str(U) + ', V=' + str(V))
    matplotlib.pyplot.show()

def compute_intersect_values(a1, b1, a2, b2):
    """Return the two values used to determine if two segments intersect."""
    U = np.dot(np.cross(np.array(b1 - a1), np.array(b2 - b1)), np.cross(np.array(b1 - a1), np.array(a2 - b1)))
    V = np.dot(np.cross(np.array(b2 - a2), np.array(b1 - b2)), np.cross(np.array(b2 - a2), np.array(a1 - b2)))
    return U, V


def is_inside_form(node_coords, borders, point_coords):
    """Returns True if the point of coordinates points_coords = np.array([x,y]) is inside the form
    described by borders = np.array([id1, id2, id3], [id4, id5])"""
    x_max = node_coords[borders[0][0]][0]
    for i in range(len(borders)):
        for j in range(len(borders[i])):
            if is_in_segment(node_coords[borders[i][j]], node_coords[borders[i][(j+1)%len(borders[i])]], point_coords):
                return True
            if point_coords[0] == node_coords[borders[i][j]][0] and point_coords[1] == node_coords[borders[i][j]][1]:
                return True
            elif x_max < node_coords[borders[i][j]][1]:
                x_max = node_coords[borders[i][j]][1]
    x_max += 1
    C = 0
    e_fictive = np.array([x_max, point_coords[1] + 0.9382034]) # random number to avoid parallelness
    E = 0
    i = 0
    while i < len(borders):
        j = 0
        while j < len(borders[i]):
            U, V = compute_intersect_values(e_fictive, point_coords, node_coords[borders[i][j]], node_coords[borders[i][(j+1)%len(borders[i])]])
            if U == 0 and V == 0:
                e_fictive[0] = e_fictive[0] +  0.69 # random number to avoid parallelness
                e_fictive[1] = e_fictive[1] +  0.283 # random number to avoid parallelness
                i, j = 0, 0
                C = 0
            else:
                if U < 0 and V < 0:
                    C += 1
                j += 1
                E += 1
        i += 1
    return C%2 == 1

def _test_is_inside_form():
    node_coords = np.array([[0, 0], [0, 3], [3, 3], [3, 0], [1, 1], [2, 1], [2, 2], [1, 2]])
    p_elem2nodes = np.array([0, 4, 8])
    elem2nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    assert is_inside_form(node_coords, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), np.array([1.5, 1.5])) == False, "Error in the test of is_inside_form"
    assert is_inside_form(node_coords, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), np.array([2, 1.5])) == True, "Error in the test of is_inside_form"
    assert is_inside_form(node_coords, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), np.array([2.5, 1.5])) == True, "Error in the test of is_inside_form"


def apply_non_convex_Delaunay_triangulation(point_coords, borders):
    """Create a Delaunay triangulation with no element outsite the form
    limited by the borders"""
    nodesid = np.array([], np.int64)
    for i in range(len(borders)):
        for j in range(len(borders[i])):
            nodesid = np.append(nodesid, borders[i][j])
    node_coords, p_elem2nodes, elem2nodes = apply_Delaunay_triangulation(point_coords, nodesid)
    elem_to_delete = np.array([], dtype=np.int64)
    for elemid in range(len(p_elem2nodes) - 1):
        barycenter_coords = compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes, elemid)
        if not is_inside_form(node_coords, borders, barycenter_coords) and elemid not in elem_to_delete:
            elem_to_delete = np.append(elem_to_delete, elemid)
    for elemid in np.sort(elem_to_delete)[::-1]:
        node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid)

    return node_coords, p_elem2nodes, elem2nodes


def _test_apply_non_convex_Delaunay_triangulation():
    motif = [[0.25, 0], [0.25, 0.25], [0.5, 0.25], [0.5, 0], [0.5, -0.25], [0.75, -0.25], [0.75, 0]]
    point_coords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    border = np.array([0, 1, 2, 3, 0])
    point_coords, border = generate_fractal_border(point_coords, border, motif, 2)
    border = border[:-1]
    node_coords, p_elem2nodes, elem2nodes = apply_non_convex_Delaunay_triangulation(point_coords, np.array([border]))
    plot_all_elem(node_coords, p_elem2nodes, elem2nodes)
    plot_all_elem(node_coords, np.array([0, len(border)]), border, colorname="red")
    matplotlib.pyplot.title('Test for applying non convex Delaunay triangulation')
    matplotlib.pyplot.show()


if __name__ == '__main__':
    # _test_is_in_circumcenter()
    # _test_generate_random_point_uniformly()
    # _test_is_counterclockwise()
    # _test_construct_fictive_triangle()
    # _test_add_one_point_to_triangulation()
    # _test_apply_Delaunay_triangulation()
    # _test_do_intersect()
    # _test_is_inside_form()
    _test_apply_non_convex_Delaunay_triangulation()
    pass