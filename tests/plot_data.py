#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

file = h5.File("SeisCL_dout.mat", "r")
clip=0.01
vmax = clip*np.max(file["pout"])
vmin=-vmax

plt.imshow(file["pout"], aspect="auto", vmin=vmin, vmax=vmax)
plt.show()
file.close()
