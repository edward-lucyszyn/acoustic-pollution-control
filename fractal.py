# -*- coding: utf-8 -*-

# Python packages
import numpy as np
import time

# Files import
from plot_functions import *
from mesh_control import *


def compute_adjusted_coords(x_a, y_a, x_b, y_b, x, y):
    """Operate a translation, an homotethy and a rotation to compute 
    the coordinates of x and y as if x_a, ya = 0, 0 and x_b, y_b = 1, 0."""
    c = np.sqrt((x_b - x_a)**2 + (y_b - y_a)**2)
    x, y = c*x, c*y
    if x_b > x_a: theta = np.arctan((y_b - y_a)/(x_b - x_a))
    elif x_a > x_b: theta = np.pi + np.arctan((y_b - y_a)/(x_b - x_a))
    elif x_a == x_b and y_b > y_a: theta = np.pi/2
    else: theta = 3*(np.pi/2)
    x, y = x*np.cos(theta) - y*np.sin(theta), x*np.sin(theta) + y*np.cos(theta)
    x, y = x + x_a, y + y_a
    return x, y


def compute_side_coords(x_a, y_a, x_b, y_b, motif_x, motif_y):
    """Return the coordinates of the point in the pattern with the appropriate coordinates 
    respectively to the side that we considerate. This side is represented with points of 
    coordinates x_a, y_a and x_b, y_b."""
    X_array = np.array([])
    Y_array = np.array([])
    for i in range(len(motif_x)):
        x, y = compute_adjusted_coords(x_a, y_a, x_b, y_b, motif_x[i], motif_y[i])
        X_array, Y_array = np.append(X_array, x), np.append(Y_array, y)
    return X_array, Y_array


def one_iteratation_fractal(node_coords, p_elem2nodes, elem2nodes, border, motif_x, motif_y):
    """Iterate the fractal one time."""
    new_border = np.array([], dtype=np.int64)
    for i in range(len(border) - 1):
        new_border = np.append(new_border, border[i])
        X_array, Y_array = compute_side_coords(node_coords[border[i]][0], node_coords[border[i]][1], node_coords[border[i+1]][0], node_coords[border[i+1]][1], motif_x, motif_y)
        for j in range(len(X_array)):
            node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, np.array([[X_array[j], Y_array[j]]]))
            new_border = np.append(new_border, len(node_coords) - 1)
    new_border = np.append(new_border, border[-1])
    return node_coords, new_border


def show_fractal(node_coords, p_elem2nodes, elem2nodes, border, motif, name, colors_dic = {3: 'orange'}, dpi=500):
    """Generate a fractal and draw each step with the color associated in colors_dic. For example, colors_dic = {3 : 'purple'}."""
    start = time.time()

    motif_x = np.array([element[0] for element in motif])
    motif_y = np.array([element[1] for element in motif])

    targetted_i = list(colors_dic.keys())
    i_max = max(targetted_i)
    for i in range(i_max + 1):
        if i in targetted_i:
            matplotlib.pyplot.plot((node_coords[border[0]][0], node_coords[border[1]][0]), (node_coords[border[0]][1], node_coords[border[1]][1]), colors_dic[i], label='n=' + str(i))
            for j in range(1, len(border) - 1):
                matplotlib.pyplot.plot((node_coords[border[j]][0], node_coords[border[j+1]][0]), (node_coords[border[j]][1], node_coords[border[j+1]][1]), colors_dic[i])
        node_coords, border = one_iteratation_fractal(node_coords, p_elem2nodes, elem2nodes, border, motif_x, motif_y)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(str(name) + " n=" + str(i_max) + str(" dpi=" + str(dpi)), dpi=500)
    end = time.time()
    print('\n * Execution finished in ' + str(int(end - start)) + 's. *')


def generate_fractal_border(point_coords, border, motif, n):
    """Returns the list of coordinates and the succession """
    p_elem2nodes = np.array([0])
    elem2nodes = np.array([])
    motif_x = np.array([element[0] for element in motif])
    motif_y = np.array([element[1] for element in motif])
    for _ in range(n):
        point_coords, border = one_iteratation_fractal(point_coords, p_elem2nodes, elem2nodes, border, motif_x, motif_y)
    return point_coords, border


if __name__ == '__main__':
    motif = [[0.25, 0], [0.25, 0.25], [0.5, 0.25], [0.5, 0], [0.5, -0.25], [0.75, -0.25], [0.75, 0]]
    # motif = [[0.25, 0], [0.25, 0.2], [0.5, 0.2], [0.5, 0], [0.5, -0.2], [0.75, -0.2], [0.75, 0]]
    # motif = [[0.25, 0], [0.375, 0.25], [0.5, 0], [0.625, -0.25], [0.75, 0]]
    trace = np.append(np.array([[0, 0]]), motif, axis=0)
    trace = np.append(trace, [[1, 0]], axis=0)

    # matplotlib.pyplot.plot([trace[i][0] for i in range(len(trace))], [trace[i][1] for i in range(len(trace))])
    # for i in range(len(motif)):
    #     matplotlib.pyplot.plot(motif[i][0], motif[i][1], 'bo', color='red')
    # matplotlib.pyplot.show()
    

    point_coords = np.array([
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        ])
    # point_coords = np.array([
    #     [0, 0],
    #     [1,0],
    #     [0.5, np.sqrt(3)/2],
    # ])

    border = np.array([0, 1, 2, 3, 0])
    # border = np.array([0, 1, 2, 0])

    elem2nodes = np.array([])
    p_elem2nodes = np.array([0])

    show_fractal(point_coords, p_elem2nodes, elem2nodes, border, motif, "Testekd,ekd,", colors_dic = {0: "yellow", 1: 'orange', 2: 'red', 3:'purple'}, dpi=300)
    point_coords, border = generate_fractal_border(point_coords, border, motif, 2)
    pass
