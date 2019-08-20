#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

file = h5.File("SeisCL_movie.mat", "r")
mov = file['movvx']
print(np.max(mov))

plt.imshow(np.transpose(mov[0,30,:,:]))
plt.show()

fig = plt.figure()
ims = []
for ii in range(mov.shape[1]):
    im = plt.imshow(np.transpose(mov[0,ii,:,:]), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

plt.show()
file.close()

