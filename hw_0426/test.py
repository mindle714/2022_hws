import numpy as np
def softmax(x):
  z = x - np.max(x, -1, keepdims=True)
  num = np.exp(z)
  denom = np.sum(num, -1, keepdims=True)
  return num / denom

dim = 16; timestep = 10
s = np.ones((dim,))
h = np.ones((timestep, dim))
mask = np.array([1.,1.,1.,1.,1.,0.,0.,0.,0.,0.])

print(softmax(np.matmul(s, h.T)))
print(softmax(np.matmul(s, h.T) + -1e9 * (1-mask)))
