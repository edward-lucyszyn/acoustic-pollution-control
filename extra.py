# -*- coding: utf-8 -*-

# Python modules
import numpy as np

def order_normal_list_with_numpy_numbers(L):
    """Returns the sorted L whose length is 2."""
    if L[0] < L[1]:
        return L
    else:
        return [L[1], L[0]]
    

def triangle_area(point_A, point_B, point_C):
    """Returns the area of a triangle thanks to Heron's formula. point_A = np.array([x, y])."""
    c = np.linalg.norm(point_A - point_B, 2)
    a = np.linalg.norm(point_B - point_C, 2)
    b = np.linalg.norm(point_C - point_A, 2)
    s = (a+b+c)/2
    return np.sqrt(s*(s -a)*(s - b)*(s - c)) #Heron's formula


def triangle_angles(point_A, point_B, point_C):
    """Returns the list of three angles in radians of the triangle."""
    c = np.linalg.norm(point_A - point_B, 2)
    a = np.linalg.norm(point_B - point_C, 2)
    b = np.linalg.norm(point_C - point_A, 2)
    alpha = np.arccos((b**2 + c**2 - a**2)/(2*b*c))
    beta = np.arccos((a**2 + c**2 - b**2)/(2*a*c))
    gamma = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    return np.array([alpha, beta, gamma])


def triangle_angles_in_degrees(point_A, point_B, point_C):
    """Returns the list of three angles in degrees of the triangle."""
    return triangle_angles(point_A, point_B, point_C)*(180/np.pi)


def is_in_segment(a, b, c, epsilon = 1e-10):
    """Returns True if point c belongs to segment [a, b]. a = np.array([x, y])."""
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True


if __name__ == "__main__":
    print('\n Test for triangle_area:')
    print(triangle_area(np.array([0, 0]), np.array([1, 2]), np.array([1, 0])))
    print('\n Test for triangle angles in degrees')
    print(triangle_angles_in_degrees(np.array([0, 0]), np.array([1, 2]), np.array([1, 0])))
    assert is_in_segment(np.array([0, 0]), np.array([0, 1]), np.array([0, 2])) == False, "Error in is_in_segment"
    assert is_in_segment(np.array([0, 0]), np.array([0, 2]), np.array([0, 1])) == True, "Error in is_in_segment"
    assert is_in_segment(np.array([0, 0]), np.array([2, 2]), np.array([0, 0])) == True, "Error in is_in_segment"
    assert is_in_segment(np.array([0, 0]), np.array([2, 4]), np.array([1, 2])) == True, "Error in is_in_segment"
    pass
