#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

file = h5.File("./seiscl/SeisCL_movie.mat", "r")
mov = file['movvx']
print(np.max(mov))

slices = [d//2 for d in file['movvx'].shape[2:]]
slices[0] = slice(16, -16)
slices[-1] = slice(16, -16)

plt.imshow(np.transpose(mov[tuple([0,1]+slices)]))
plt.show()

fig = plt.figure()
ims = []
for ii in range(mov.shape[1]):
    im = plt.imshow(np.transpose(mov[tuple([0,ii]+slices)]), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

plt.show()
file.close()

