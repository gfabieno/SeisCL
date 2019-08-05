#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

file = h5.File("SeisCL_dout.mat", "r")
file2 = h5.File("SeisCL_din.mat", "r")
clip=1.0
vmax = clip*np.max(file["pout"])
vmin=-vmax

plt.imshow(file["pout"][:]-file2["p"][:], aspect="auto")
plt.show()
file.close()
file2.close()
