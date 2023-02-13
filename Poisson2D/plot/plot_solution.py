#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import sys

if(len(sys.argv) > 2):
    print('Too many arguments')
    print('Correct usage: python plot_solution.py file.csv')
    exit()
if(len(sys.argv) < 2):
    print('Too few arguments')
    print('Correct usage: python plot_solution.py file.csv')
    exit()

points = pd.read_csv(sys.argv[1], sep=',')

fig = plt.figure();
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('v')

x = points['x'].values
y = points['y'].values
v = points['v'].values

ax.plot_trisurf(x,y,v, cmap=cm.jet)

# NX = 1024
# NY = 16

# X = np.linspace(0,64,NX)
# Y = np.linspace(0,1,NY)
# X,Y = np.meshgrid(X,Y)
# Z = 8000*np.sin(2 * X / (64) * np.pi)

# ax.plot_surface(X,Y,Z, cmap=cm.jet)




plt.show()

