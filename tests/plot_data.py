#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt


file = h5.File("SeisCL_dout.mat", "r")
plt.imshow(file["pout"], aspect="auto")
plt.show()
file.close()
