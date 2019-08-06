#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

file = h5.File("SeisCL_dout.mat", "r")
clip=1.0
vmax = clip*np.max(file["vxout"])
vmin=-vmax

plt.imshow(file["vxout"][:], aspect="auto")
plt.show()
file.close()

