"""
Project 3
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 #run pip install opencv-python
import time 
from random import randint

PGDIM = [1000,1000,1000] #Plagground Dimensions [x,y,z]

PG = np.zeros((PGDIM[0],PGDIM[1],PGDIM[2]))

class walkergen(object):
    def __init__(self,DIM):
        self.x = randint(0,DIM[0]) 
        self.y = randint(0,DIM[1])
        self.z = randint(0,DIM[2])
        self.coord = [self.x,self.y,self.z]

W = {n: walkergen(PGDIM) for n in range(1000)}

