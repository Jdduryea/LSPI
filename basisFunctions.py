'''
This module just contains a bunch of basis 
functions that can be used with various problems
alongside the LSPI algorithm. Make sure the basis
function lines up with the correct problem you 
are trying to solve!
'''

import numpy as np


# USE ME FOR CARTPOLE!!
def get_cartpole_basis_functions_quadratic_v2():
    '''
    This one does really well! USE THIS ONE!!
    Just some simple quadratics
    '''

    Q1 = np.identity(5)
    Q2 = np.ones((5,5))
    Q3 = np.array([[1,1,1,1,-1],[1,1,1,-1,1],[1,1,-1,1,1],[1,-1,1,1,1],[-1,1,1,1,1]])


    v = lambda s,a: np.append(s,a)
    bf1 = lambda s,a:1
    bf2 = lambda s,a: np.dot(np.dot(v(s,a), Q1), v(s,a))
    bf3 = lambda s,a: np.dot(np.dot(v(s,a), Q2), v(s,a))
    bf4 = lambda s,a: np.dot(np.dot(v(s,a), Q3), v(s,a))
    
    bfs = [bf1, bf2, bf3, bf4]
    return bfs


def get_cartpole_basis_functions_quadratic_v0():
    Q1 = np.ones((5,5))

    v = lambda s,a: np.append(s,a)
    bf1 = lambda s,a:1
    bf2 = lambda s,a: np.dot(np.dot(v(s,a),Q1),v(s,a))
    
    bfs = [bf1,bf2]
    return bfs

def get_cartpole_basis_functions_quadratic_v1():
    Q1 = np.identity(5)
    Q2 = np.ones((5,5))

    v = lambda s,a: np.append(s,a)
    bf1 = lambda s,a:1
    bf2 = lambda s,a: np.dot(np.dot(v(s,a),Q1),v(s,a))
    bf3 = lambda s,a: np.dot(np.dot(v(s,a),Q2),v(s,a))
    
    bfs = [bf1,bf2, bf3]
    return bfs



def get_cartpole_basis_functions_quadratic_v3():
    '''
    This one just moves the car to the left
    '''

    Q1 = np.array([[1,1,1,1,-1],[1,1,1,-1,1],[1,1,-1,1,1],[1,-1,1,1,1],[-1,1,1,1,1]])

    v = lambda s,a: np.append(s,a)
    bf1 = lambda s,a:1
    bf2 = lambda s,a: np.dot(np.dot(v(s,a),Q1),v(s,a))
    
    bfs = [bf1,bf2]
    return bfs

def get_cartpole_basis_functions_v1():
    bf1 = lambda s,a:a
    bf2 = lambda s,a:s[0]
    bf3 = lambda s,a:s[1]
    bf4 = lambda s,a:s[2]
    bf5 = lambda s,a:s[3]
    bfs = [bf1,bf2,bf3,bf4,bf5]
    return bfs
   
def get_cartpole_basis_functions_v2():
    '''
    Returns a list of basis functions that seem 
    to work well for the simplified (no singularity, starting above the horizon)
    cartpole problem.
    
    '''
    s1 = np.array([1,1,1,1])
    s2 = np.array([0,0,0,0])
    s3 = np.array([1,0,1,0])

    bf1 = lambda s,a: 1

    bf2 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s1)/2.0)
    bf3 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s1)/2.0)

    bf4 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s2)/2.0)
    bf5 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s2)/2.0)

    bf6 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s3)/2.0)
    bf7 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s3)/2.0)
    
    
    bfs = [bf1,bf2, bf3, bf4, bf5, bf6, bf7]
    
    return bfs


def get_cartpole_basis_functions_v3():

    s1 = np.array([-1,-1,0,0])
    s2 = np.array([-0.5,-1,0,0])
    s3 = np.array([-0.1,-1,0,-0.5])
    s4 = np.array([0,0,0,0])
    s5 = np.array([0.1,1,0,0])
    s6 = np.array([0.5,0.5,0,-0.5])
    s7 = np.array([1,0,0,0])

    bf1 = lambda s,a: 1

    bf2 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s1)/2.0)
    bf3 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s1)/2.0)

    bf4 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s2)/2.0)
    bf5 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s2)/2.0)

    bf6 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s3)/2.0)
    bf7 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s3)/2.0)
    
    bf8 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s4)/2.0)
    bf9 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s4)/2.0)
    
    bf10 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s5)/2.0)
    bf11 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s5)/2.0)
    
    bf12 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s6)/2.0)
    bf13 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s6)/2.0)
    
    bf14 = lambda s,a: int(a==0)*np.exp( - np.linalg.norm(s-s7)/2.0)
    bf15 = lambda s,a: int(a==1)*np.exp( - np.linalg.norm(s-s7)/2.0)

    bfs = [bf1,bf2,bf3,bf4, bf5, bf6, bf7, bf8, bf9, bf10, bf11, bf12, bf13, bf14, bf15]
    return bfs
    

def get_quadratic_bfs ():
    Q1 = np.identity(3)
    Q2 = np.array([[1,2,3],[3,2,1],[1,3,2]])
    Q3 = np.array([[-3,1,2],[-3,1,-2],[-1,3,2]])
    Q4 = np.array([[1,-2,-1],[1,-1,2],[1,-3,-2]])
    v = lambda s,a: np.append(s,a) # short hand
    bf1 = lambda s,a: np.dot(np.dot(v(s,a).T, Q1), v(s,a))
    bf2 = lambda s,a: np.dot(np.dot(v(s,a).T, Q2), v(s,a))
    bf3 = lambda s,a: np.dot(np.dot(v(s,a).T, Q3), v(s,a))
    bf4 = lambda s,a: np.dot(np.dot(v(s,a).T, Q4), v(s,a))

    return [bf1,bf2,bf3, bf4]
    
def get_cartpole_basis_functions_v4():
    bf1 = lambda s,a: 1
    bf2 = lambda s,a: a
    bf3 = lambda s,a: int(a==0)*s[0]
    bf4 = lambda s,a: int(a==0)*s[1]
    bf5 = lambda s,a: int(a==0)*s[2]
    bf6 = lambda s,a: int(a==0)*s[3]
    bf7 = lambda s,a: int(a==1)*s[0]
    bf8 = lambda s,a: int(a==1)*s[1]
    bf9 = lambda s,a: int(a==1)*s[2]
    bf10 = lambda s,a: int(a==1)*s[3]
    return [bf1,bf2,bf3,bf4,bf5, bf6, bf7, bf8, bf8, bf10]


def get_mt_car_basis_functions_quadratic_v1():
    Q1 = np.identity(3)
    Q2 = np.ones((3,3))
    Q3 = np.array([[1,1,-1],[1,-1,1],[-1,1,1]])

    v = lambda s,a: np.append(s,a)
    bf1 = lambda s,a:1
    bf2 = lambda s,a: np.dot(np.dot(v(s,a),Q1),v(s,a))
    bf3 = lambda s,a: np.dot(np.dot(v(s,a),Q2),v(s,a))
    bf4 = lambda s,a: np.dot(np.dot(v(s,a),Q3),v(s,a))
    
    bfs = [bf1,bf2, bf3, bf4]
    return bfs

    
def get_continuous_mt_car_basis_functions():
    '''
    Define some basis functions and return them in a list
    '''

    bf1 = lambda s,a: 1
    bf2 = lambda s,a: s[0]
    bf3 = lambda s,a: a
    bf4 = lambda s,a: a*s[0]*s[0]
    bf5 = lambda s,a: a*s[0]*s[0]*s[0]
    bf6 = lambda s,a: a*s[1]
    bf7 = lambda s,a: a*s[1]*s[1]
    bf8 = lambda s,a: a*s[1]*s[1]*s[1]
    bf9 = lambda s,a: a*s[0]*s[1]
    bf10 = lambda s,a: a*s[0]*s[0]*s[1]*s[1]

    return [bf1,bf2,bf3,bf4,bf5,bf6,bf7,bf8,bf9,bf10]

def get_non_linear_mt_car_basis_functions():
    '''
    Define some basis functions and return them in a list
    '''

    bf1 = lambda s,a: 1
    bf2 = lambda s,a: s[0]
    bf3 = lambda s,a: a
    bf4 = lambda s,a: (a**5 + a**2 -a)*s[0]*s[0]
    bf5 = lambda s,a: (a**5 + a**2 -a)*s[0]*s[0]*s[0]
    bf6 = lambda s,a: (a**5 + a**2 -a)*s[1]
    bf7 = lambda s,a: (a**5 + a**2 -a)*s[1]*s[1]
    bf8 = lambda s,a: (a**5 + a**2 -a)*s[1]*s[1]*s[1]
    bf9 = lambda s,a: (a**5 + a**2 -a)*s[0]*s[1]
    bf10 = lambda s,a: (a**5 + a**2 -a)*s[0]*s[0]*s[1]*s[1]

    return [bf1,bf2,bf3,bf4,bf5,bf6,bf7,bf8,bf9,bf10]


def get_acrobat_basis_functions_quadratic_v1():
    '''
    This one just moves the car to the left
    '''
    Q1 = np.identity(5)
    Q2 = np.ones((5,5))
    Q3 = np.array([[1,1,1,1,-1],[1,1,1,-1,1],[1,1,-1,1,1],[1,-1,1,1,1],[-1,1,1,1,1]])
    v = lambda s,a: np.append(s[-4:],a)

    bf1 = lambda s,a:1
    bf2 = lambda s,a: np.dot(np.dot(v(s,a),Q1),v(s,a))
    bf3 = lambda s,a: np.dot(np.dot(v(s,a),Q2),v(s,a))
    bf4 = lambda s,a: np.dot(np.dot(v(s,a),Q3),v(s,a))
    
    bfs = [bf1,bf2, bf3, bf4]
    return bfs


