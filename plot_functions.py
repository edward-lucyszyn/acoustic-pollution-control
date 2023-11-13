# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot


def plot_elem(node_coords, p_elem2nodes, elem2nodes, elem, colorname='orange'):
    xyz = node_coords[ elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
    for i in range(len(xyz) - 1):
        matplotlib.pyplot.plot((xyz[i,0], xyz[i+1,0]), (xyz[i,1], xyz[i+1,1]), color=colorname)
    matplotlib.pyplot.plot((xyz[len(xyz) - 1,0], xyz[0,0]), (xyz[len(xyz) - 1,1], xyz[0,1]), color=colorname)
    return


def plot_all_elem(node_coords, p_elem2nodes, elem2nodes, colorname='orange'):
    for i in range(len(p_elem2nodes) - 1):
        plot_elem(node_coords, p_elem2nodes, elem2nodes, i, colorname=colorname)
    return


def plot_node(node_coords, p_elem2nodes, elem2nodes, node, colorname='red'):
    matplotlib.pyplot.plot(node_coords[node, 0], node_coords[node, 1], 'bo', color=colorname)
    return


def plot_all_node(node_coords, p_elem2nodes, elem2nodes, colorname='red'):
    for i in range(len(node_coords)):
        plot_node(node_coords, p_elem2nodes, elem2nodes, i, colorname)
    return
