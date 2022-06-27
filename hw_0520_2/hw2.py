import numpy as np

def _fwht(a) -> None:
  """In-place Fast Walsh–Hadamard Transform of array a."""
  h = 1
  while h < len(a):
    for i in range(0, len(a), h * 2):
      for j in range(i, i + h):
        x = a[j]
        y = a[j + h]
        a[j] = x + y
        a[j + h] = x - y
    h *= 2

#a = np.array([
#  [4,3,2,1],
#  [3,2,1,1],
#  [2,1,1,1],
#  [1,1,1,1]
#], dtype=np.float32)
a = np.array([
  [4,3,2,1],
  [5,2,1,1],
  [6,8,1,1],
  [7,9,2,1]
], dtype=np.float32)

ac = np.copy(a)
for idx in range(ac.shape[0]):
  _fwht(ac[idx])

print(ac)
ac = ac.T
#ac = np.copy(a)
for idx in range(ac.shape[0]):
  _fwht(ac[idx])
ac = ac.T
print(ac)

wht4 = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], dtype=np.float32)
wht4 *= 0.5
#print(wht4)
print(a)

print("line37"); print(np.matmul(a, wht4))
print(np.matmul(np.matmul(a, wht4).T, wht4).T)

print("line39"); print(np.matmul(a.T, wht4))
print(np.matmul(np.matmul(a.T, wht4).T, wht4))

'''
from sympy import fwht
print(fwht(np.array(fwht(a.flatten())).reshape([4,4]).T.flatten()))
print(fwht(np.array(fwht(a.T.flatten())).reshape([4,4]).flatten()))

print(np.array(fwht(a.flatten())).reshape([4,4]))
'''
