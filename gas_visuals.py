# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:35:01 2026

@author: pelgr
"""
import matplotlib.pyplot as plt
import numpy as np
from gas_visualizer import GasVisualizer

box_size = 100
radius = 2
mass = 1

# create starting positions of 100 particles
x_array = np.linspace(-45, 45, 10)
y_array = np.linspace(-45, 45, 10)
r = np.zeros((100, 2))

for i, x in enumerate(x_array):
    for j, y in enumerate(y_array):
        r[10 * i:10 * (1+ i),0] = x
        r[j + 10*i, 1] = y

r = r.astype(dtype='float64')

N = len(r)

# fill the circles on the right side
fill = []
for x in r[:,0]:
    if x < 0:
        fill.append(True)
    elif x > 0:
        fill.append(False)


# create random speeds with the same total speed but different direction
rng = np.random.default_rng()
v = np.zeros((100, 2))

for i in range(N):
    v_x = rng.uniform(-1.5,1.5)
    v[i,0] = v_x
    v_y = np.sqrt(2.25 - v_x**2)
    plus_minus = int(np.random.uniform(1, 3))
    if plus_minus == 1:
        v_y = -v_y
    v[i,1] = v_y

MSDs = []
times = []
dt = 1

def update_simulation():
    resolve_collisions(r, v, radius)
    wall_collisions(r, v, radius, box_size)
    MSDs.append(diffusion(r))
    r[:] += v * dt
    
    
    
    
def wall_collisions(r, v, radius, box_size):
    for i, n in enumerate(r):
        if n[0] >= ((box_size / 2) - radius) or n[0] <= ((-box_size / 2) + radius):
            v[i, 0] = -v[i, 0]
        if n[1] >= ((box_size / 2) - radius) or n[1] <= ((-box_size / 2) + radius):
            v[i, 1] = -v[i, 1]
    return v
    

def resolve_collisions(r, v, radius):
    '''
    function for checking and resolving colissions between 2 particles
    '''
    
    # alle combinaties van de deeltjes
    for i, r1 in enumerate(r):
        for j, r2 in enumerate(r):
            if j < i + 1:
                continue
            
            pre_distance = r1 - r2
            distance = np.sqrt(pre_distance[0] ** 2 + pre_distance[1] ** 2)
            if distance <= (2 * radius):
                if np.dot((r1 - r2), (v[i] - v[j])) < 0:
                    v1 = v[i] * 1       # de keer 1 zorgt ervoor dat v1 niet wordt verandert
                    v2 = v[j] * 1       # als v[i] wordt verandert, hetzelfde voor v2 en v[j]
                    v[i] = v1 - (pre_distance * (np.dot((v1 - v2), (pre_distance))) / (distance ** 2))
                    v[j] = v2 + (pre_distance * (np.dot((v2 - v1), (-pre_distance))) / (distance ** 2))
    return v

def diffusion(r):
    distances = []
    for i, r1 in enumerate(r):
        for j, r2 in enumerate(r):
            if j < i + 1:
                continue
            
            pre_distance = r1 - r2
            distance = (np.sqrt(pre_distance[0] ** 2 + pre_distance[1] ** 2))
            distances.append(distance ** 2)
    MSD = np.mean(distances)
    return MSD
    
    
def total_energy(v, mass, N):
    et = 0
    for i in range(N):
        ek = 0.5 * mass * (v[i,0] ** 2 + v[i, 1] ** 2)
        et += ek
    return et
for i in range(100):
    t = i * dt
    times.append(t)
visualizer = GasVisualizer(box_size=box_size, radius=radius)
print(total_energy(v, mass, N))
visualizer.run(r, update_simulation, filled=fill)
print(total_energy(v, mass, N))
plt.plot(times ,MSDs)