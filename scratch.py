import numpy as np

a = np.ones((1,3))
b = np.ones((1,3))

np.save("rand", [a,b])

c = np.load("rand.npy")
print(type(c))