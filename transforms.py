"""
Transforms Module - Contains code for to learn about rotations
and eventually homogenous transforms. 

Empty outline derived from code written by John Morrell. 
"""

import numpy as np
from numpy import sin, cos, sqrt
from numpy.linalg import norm
import sympy as sp

## 2D Rotations
def rot2(th):
    """
    R = rot2(theta)
    Parameters
        theta: float or int, angle of rotation
    Returns
        R: 2 x 2 numpy array representing rotation in 2D by theta
    """

    R = np.array([[cos(th), -sin(th)], [sin(th), cos(th)]])
    return R

## 3D Transformations
def rotx(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about x-axis by amount theta
    """

    R = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])

    return R

def roty(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about y-axis by amount theta
    """

    R = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])
 
    return R

def rotz(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about z-axis by amount theta
    """

    R = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])

    return R

# inverse of rotation matrix 
def rot_inv(R):
    '''
    R = rot_inv(R)
    Parameters
        R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    Returns
        R: 2x2 or 3x3 inverse of the input rotation matrix
    '''
    
    return np.transpose(R)



''' SYMPY TRANSFORMS'''

## 2D Rotations
def sp_rot2(th):
    """
    R = rot2(theta)
    Parameters
        theta: float or int, angle of rotation
    Returns
        R: 2 x 2 numpy array representing rotation in 2D by theta
    """

    R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    return R

## 3D Transformations
def sp_rotx(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about x-axis by amount theta
    """

    R = sp.Matrix([[1, 0, 0], [0, sp.cos(th), -sp.sin(th)], [0, sp.sin(th), sp.cos(th)]])

    return R

def sp_roty(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about y-axis by amount theta
    """

    R = sp.Matrix([[sp.cos(th), 0, sp.sin(th)], [0, 1, 0], [-sp.sin(th), 0, sp.cos(th)]])
 
    return R

def sp_rotz(th):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about z-axis by amount theta
    """

    R = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])

    return R

# inverse of rotation matrix 
def sp_rot_inv(R):
    '''
    R = rot_inv(R)
    Parameters
        R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    Returns
        R: 2x2 or 3x3 inverse of the input rotation matrix
    '''
    
    return sp.transpose(R)
def se3(R=np.eye(3), p=np.array([0, 0, 0])):
    """
        T = se3(R, p)
        Description:
            Given a numpy 3x3 array for R, and a 1x3 or 3x1 array for p, 
            this function constructs a 4x4 homogeneous transformation 
            matrix "T". 

        Parameters:
        R - 3x3 numpy array representing orientation, defaults to identity
        p = 3x1 numpy array representing position, defaults to [0, 0, 0]

        Returns:
        T - 4x4 numpy array
    """

    T = np.zeros((4, 4))
    T[:3,:3] = R
    T[:3, 3] = p
    T[3] = np.array([0, 0, 0, 1])

    return T

def inv(T):
    """
        Tinv = inv(T)
        Description:
        Returns the inverse transform to T

        Parameters:
        T

        Returns:
        Tinv - 4x4 numpy array that is the inverse to T so that T @ Tinv = I
    """

    R = T[:3,:3]
    p = T[:3, 3]
    R_inv = R.transpose()
    p_inv = R_inv @ -p
    T_inv = se3(R_inv, p_inv)

    return T_inv

"""
SO(3) conversion code to convert between different SO(3) representations. 

Copy this file into your 'transforms.py' file at the bottom. 
"""

#### HW 4 Functions ####

def R2rpy(R):
    """
    rpy = R2rpy(R)
    Description:
    Returns the roll-pitch-yaw representation of the SO3 rotation matrix

    Parameters:
    R - 3 x 3 Numpy array for any rotation

    Returns:
    rpy - 1 x 3 Numpy Matrix, containing <roll pitch yaw> coordinates (in radians)
    """
    
    # follow formula in book, use functions like "np.atan2" 
    # for the arctangent and "**2" for squared terms. 

    roll = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[2,1], R[2,2])

    return np.array([roll, pitch, yaw])

def R2axis(R):
    """
    axis_angle = R2axis(R)
    Description:
    Returns an axis angle representation of a SO(3) rotation matrix

    Parameters:
    R

    Returns:
    axis_angle - 1 x 4 numpy matrix, containing  the axis angle representation
    in the form: <angle, rx, ry, rz>
    """

    # see equation (2.27) and (2.28) on pg. 54, using functions like "np.acos," "np.sin," etc. 
    ang = np.arccos((np.trace(R) - 1)/2)
    axis_angle = np.array([ ang, 
                            1/(2*np.sin(ang)) * (R[2,1] - R[1,2]),
                            1/(2*np.sin(ang)) * (R[0,2] - R[2,0]),
                            1/(2*np.sin(ang)) * (R[1,0] - R[0,1])])
                        

    return axis_angle

def axis2R(ang, v):
    """
    R = axis2R(angle, rx, ry, rz, radians=True)
    Description:
    Returns an SO3 object of the rotation specified by the axis-angle

    Parameters:
    angle - float, the angle to rotate about the axis in radians
    v = [rx, ry, rz] - components of the unit axis about which to rotate as 3x1 numpy array
    
    Returns:
    R - 3x3 numpy array
    """
    rx = v[0]
    ry = v[1]
    rz = v[2]


    # alpha = np.arcsin(ry/(np.sqrt(rx**2 + ry**2)))
    # beta = np.arcsin((np.sqrt(rx**2 + ry**2)))

    # R = rotz(alpha) @ roty(beta) @ rotz(ang) @ roty(-beta) @ rotz(-alpha)

    
    R = np.array([[rx**2 * (1 - np.cos(ang)) + np.cos(ang),
                  rx * ry * (1 - np.cos(ang)) - rz * np.sin(ang), 
                  rx * rz * (1 - np.cos(ang)) + ry * np.sin(ang)],

                 [rx * ry * (1 - np.cos(ang)) + rz * np.sin(ang),
                  ry**2 * (1 - np.cos(ang)) + np.cos(ang),
                  ry * rz * (1 - np.cos(ang)) - rx * np.sin(ang)],

                 [rx * rz * (1 - np.cos(ang)) - ry * np.sin(ang),
                  ry * rz * (1 - np.cos(ang)) + rx * np.sin(ang),
                  rz**2 * (1 - np.cos(ang)) + np.cos(ang)]])
    return R

def R2q(R):
    """
    quaternion = R2q(R)
    Description:
    Returns a quaternion representation of pose

    Parameters:
    R

    Returns:
    quaternion - 1 x 4 numpy matrix, quaternion representation of pose in the 
    format [nu, ex, ey, ez]
    """

    nu = np.sqrt(np.trace(R) + 1)/2
    e = np.array([[np.sign(R[2,1] - R[1,2]) * np.sqrt(R[0,0] - R[1,1] - R[2,2] + 1)],
                 [np.sign(R[0,2] - R[2,0]) * np.sqrt(R[1,1] - R[2,2] - R[0,0] + 1)],
                 [np.sign(R[1,0] - R[0,1]) * np.sqrt(R[2,2] - R[0,0] - R[1,1] + 1)]]) / 2

    r2q = np.zeros((4,1))
    r2q[0] = nu
    r2q[1:4] = e
    return r2q
                    
def q2R(q):
    """
    R = q2R(q)
    Description:
    Returns a 3x3 rotation matrix

    Parameters:
    q - 4x1 numpy array, [nu, ex, ey, ez ] - defining the quaternion
    
    Returns:
    R - a 3x3 numpy array 
    """
    nu = q[0]
    ex = q[1]
    ey = q[2]
    ez = q[3]
    R =  np.array([[2 * (nu**2 + ex**2) - 1,
                    2 * (ex * ey - nu * ez), 
                    2 * (ex * ez + nu * ey)], 

                   [2 * (ex * ey + nu * ez),
                    2 * (nu**2 + ey**2) - 1,
                    2 * (ey * ez - nu * ex)],

                   [2 * (ex * ez - nu * ey),
                    2 * (ey * ez + nu * ex),
                    2 * (nu**2 + ez**2) - 1]])
    return R


def euler2R(th1, th2, th3, order='xyz'):
    """
    R = euler2R(th1, th2, th3, order='xyz')
    Description:
    Returns a 3x3 rotation matrix as specified by the euler angles, we assume in all cases
    that these are defined about the "current axis," which is why there are only 12 versions 
    (instead of the 24 possiblities noted in the course slides). 

    Parameters:
    th1, th2, th3 - float, angles of rotation
    order - string, specifies the euler rotation to use, for example 'xyx', 'zyz', etc.
    
    Returns:
    R - 3x3 numpy matrix
    """

    # TODO - fill out each expression for R based on the condition 
    # (hint: use your rotx, roty, rotz functions)
    if order == 'xyx':
        R = rotx(th1) @ roty(th2) @ rotx(th3)
    elif order == 'xyz':
        R = rotx(th1) @ roty(th2) @ rotz(th3)
    elif order == 'xzx':
        R = rotx(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'xzy':
        R = rotx(th1) @ rotz(th2) @ roty(th3)
    elif order == 'yxy':
        R = roty(th1) @ rotx(th2) @ roty(th3)
    elif order == 'yxz':
        R = roty(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'yzx':
        R = roty(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'yzy':
        R = roty(th1) @ rotz(th2) @ roty(th3)
    elif order == 'zxy':
        R = rotz(th1) @ rotx(th2) @ roty(th3)
    elif order == 'zxz':
        R = rotz(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'zyx':
        R = rotz(th1) @ roty(th2) @ rotx(th3)
    elif order == 'zyz':
        R = rotz(th1) @ roty(th2) @ rotz(th3)
    else:
        print("Invalid Order!")
        return

    return R
