"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
https://github.com/scipy/scipy/blob/v1.15.2/scipy/sparse/linalg/_eigen/arpack/arpack.py#L1357

This function is a wrapper to the ARPACK [1] SSEUPD and DSEUPD functions 
which use the Implicitly Restarted Lanczos Method 
to find the eigenvalues and eigenvectors [2]

[1] ARPACK Software, opencollab/arpack-ng
[2] R. B. Lehoucq, D. C. Sorensen, and C. Yang, ARPACK USERS GUIDE: Solution of Large Scale Eigenvalue Problems by Implicitly Restarted Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

"""