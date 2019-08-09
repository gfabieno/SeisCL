#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np


file = h5.File("SeisCL_gout.mat", "r")

print(np.max(file["gradvp"]))
plt.imshow(file["gradvp"][16:-16,32,16:-16], aspect="auto")
plt.show()


print(np.max(file["gradvs"]))
plt.imshow(file["gradvs"][16:-16,32,16:-16], aspect="auto")
plt.show()


print(np.max(file["gradrho"]))
plt.imshow(file["gradrho"][16:-16,32,16:-16], aspect="auto")
plt.show()
file.close()


