"""Pure-numpy copy of the repo's degree-2 lifting (same column order as PySINDy
PolynomialLibrary(degree=2, include_bias=False)). The first n columns are the raw states,
so the projection back to the state is C = [I 0]."""
import numpy as np


def lift_poly2(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[1]
    cols = [X[:, i] for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            cols.append(X[:, i] * X[:, j])
    return np.stack(cols, axis=1)


def check_against_repo(repo_lift_fn, X_sample):
    """Call once with the repo's lifting function to prove both agree."""
    a = np.asarray(repo_lift_fn(X_sample))
    b = lift_poly2(X_sample)
    assert a.shape == b.shape and np.allclose(a, b), "lifting mismatch: fix column order"
    return True
