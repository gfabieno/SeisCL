#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np


file = h5.File("SeisCL_gout.mat", "r")

slices = [d//2 for d in file["gradvp"].shape]
slices[0] = slice(16, -16)
slices[-1] = slice(16, -16)
slices=tuple(slices)

print(np.max(file["gradvp"]))
plt.imshow(np.transpose(file["gradvp"][slices]), aspect="auto")
plt.show()


print(np.max(file["gradvs"]))
plt.imshow(np.transpose(file["gradvs"][slices]), aspect="auto")
plt.show()


print(np.max(file["gradrho"]))
plt.imshow(np.transpose(file["gradrho"][slices]), aspect="auto")
plt.show()
file.close()


