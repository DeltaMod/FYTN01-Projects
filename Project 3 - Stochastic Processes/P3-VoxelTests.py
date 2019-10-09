from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def init_matrix(N=10):
    return(np.random.choice([0,1], size=(N,N,N), p=[0.99, 0.01]))

def update_matrix(mat):
    #TODO : add anihilation
    mat_1 = np.zeros(mat.shape)
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            for z in range(mat.shape[2]):
                if mat[x,y,z] : #if it is a walker
                    direct = np.random.randint(6) #pick a random direction
                    mat_1[x,y,z] = 0 
                    if direct == 0 and x != 0 :
                        mat_1[x-1, y, z] = 1
                    elif direct == 1 and y != 0 :
                        mat_1[x, y-1, z] = 1
                    elif direct == 2 and z != 0 :
                        mat_1[x, y, z-1] = 1
                    elif direct == 3 and x != mat.shape[0] - 1:
                        mat_1[x+1, y, z] = 1
                    elif direct == 4 and y != mat.shape[1] - 1:
                        mat_1[x, y+1, z] = 1
                    elif direct == 5 and z != mat.shape[2] - 1:
                        mat_1[x, y, z+1] = 1
                    else : #it is at some edge
                        mat_1[x,y,z] = 1 #it stays there
    return(mat_1)


def plot_mat(mat, title=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(mat, edgecolor="k")
    plt.show()

A = init_matrix()
plot_mat(A)

A=update_matrix(A)
plot_mat(A)
A=update_matrix(A)
plot_mat(A)
A=update_matrix(A)
plot_mat(A)
A=update_matrix(A)
plot_mat(A)
A=update_matrix(A)
plot_mat(A)
