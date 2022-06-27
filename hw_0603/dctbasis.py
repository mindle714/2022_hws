import numpy as np

def dct2basis(N):
  I, J = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, N-1, N)) 
  A = np.sqrt(2/N)*np.cos(((2.*I)*(J-1)*np.pi)/(N*2));
  A[0, :] = A[0, :] / np.sqrt(2);
  A = A.T

  B = np.zeros((N, N, N, N))
  for i in range(N):
    for j in range(N):
      B[:, :, i, j] = np.matmul(A[:, i], A[:,j].T)
  A = B
  return A

A = dct2basis(8)

import matplotlib.pyplot as plt
fig = plt.figure()

for i in range(8):
  for j in range(8):

ax = fig.add_subplot(1, 1, 1)
ax.set_axis_off()
ax.imshow(A[0][0], cmap='gray')

plt.savefig('tmp2.png')
