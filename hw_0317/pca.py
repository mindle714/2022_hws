import numpy as np
np.set_printoptions(3, suppress=True)

dim_out = 3
X = np.random.uniform(size=(10,7))

import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=dim_out)
x_pca = pca.fit_transform(X)
print(x_pca)

X = (X - np.mean(X, axis=0))
X_cov = np.cov(X, rowvar=False)
e_val, e_vec = np.linalg.eig(X_cov)
idx = np.argsort(e_val)[::-1]
e_val = e_val[idx]
e_vec = e_vec[:, idx]

print(np.dot(X, e_vec)[:,:dim_out])
