import numpy as np

arr = np.load('data/combined_images/6010_1_2.npz')

for value in arr.values():
    print(value.shape)