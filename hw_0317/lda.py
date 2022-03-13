import numpy as np
np.set_printoptions(3, suppress=True)

dim_out = 3
X = np.random.uniform(size=(20,7))
y = np.random.randint(low=1, high=7, size=20)
classes, cnt = np.unique(y, return_counts=True)
# ensure that at least 2 samples per class
while np.min(cnt) < 2:
  y = np.random.randint(low=1, high=7, size=20)
  classes, cnt = np.unique(y, return_counts=True)

if (len(classes) - 1) < dim_out:
  print("# of classes[{}] must be bigger than output dim[{}]".format(len(classes), dim_out))

import sklearn.discriminant_analysis
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=dim_out, solver='eigen')
x_lda = lda.fit_transform(X, y)
print(x_lda)

X_norm = (X - np.mean(X, axis=0))

withins = []
for i in classes:
  X_i = X_norm[y == i]
  X_i_cov = np.cov(X_i, rowvar=False, bias=True)
  withins.append(X_i_cov * (np.sum(y==i) / float(X.shape[0])))
withins = sum(withins)

X_cov = np.cov(X_norm, rowvar=False, bias=True)
betweens = X_cov - withins
#print(withins)
#print(betweens)

import scipy.linalg
#e_val, e_vec = scipy.linalg.eig(withins, betweens)
e_val, e_vec = scipy.linalg.eigh(betweens, withins)
idx = np.argsort(e_val)[::-1]
e_val = e_val[idx]
e_vec = e_vec[:, idx]
e_vec /= np.linalg.norm(e_vec, axis=0)

print(np.dot(X, e_vec)[:,:dim_out])
