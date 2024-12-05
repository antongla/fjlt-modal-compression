import numpy as np


def inst_freq(xx, dt):
    """
    This routine computes the instantaneous frequency by spectral estimation.
    Uses multiple signal classification algorithm.
    Input:
        xx - signal
        dt - time-step
    """
    xx = np.squeeze(np.asarray(xx))
    f_inst = np.zeros(len(xx))

    # PARAMETERS
    # Search for 2 complex frequencies
    P = 2
    # Window length (correlation length)
    M = 6
    # Length of the local signal
    wi = 15

    f_inst[: wi - 1] = np.NaN

    # Scan through the signal
    for i in range(wi - 1, len(xx)):
        # Cut the local signal
        x = xx[i - (wi - 1) : i + 1]
        # Generate embedded data matrix
        N = len(x) - M + 1
        X = np.zeros((N, M))
        for n in range(M):
            indList = [i + M - n - 1 for i in range(N)]
            X[:, n] = x[indList]

        # Compute dominant (output) subspace of X
        U, D, Vh = np.linalg.svd(X, full_matrices=False)
        V = Vh.conj().T

        # Spectral estimate (averaged Pisarenko)
        A = np.zeros((2 * M - 1))
        for n in range((M - P)):
            A = A + np.convolve(V[:, M - n - 1], np.conj(V[M - 1 :: -1, M - n - 1]))

        r_A = np.roots(A)
        i_min = np.where(np.abs(r_A) < 1)
        r_A_min = r_A[i_min[0]]
        freq = np.angle(r_A_min) / (2 * np.pi)

        # r_A_order = (abs(abs(r_A_min) - 1)).sort(axis=1)
        index = (abs(abs(r_A_min) - 1)).argsort()

        f_inst[i] = np.max(freq[index[0:P]]) / dt

    # Process before returning
    f_inst = 2 * np.pi * f_inst
    return f_inst[f_inst == f_inst]
