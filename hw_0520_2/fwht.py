import numpy as np

def fwht(a) -> None:
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

a = np.array([[4,3,2,1],[3,2,1,1],[2,1,1,1],[1,1,1,1]], dtype=np.float32)
wht4 = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], dtype=np.float32)
wht4 *= 0.5
print(wht4)
print(np.matmul(a, wht4))
