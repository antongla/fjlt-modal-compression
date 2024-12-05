import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
import numpy.random as npr
from numpy.random import default_rng
from math import log, sqrt
import time

from pydmd import SpDMD, DMD


def readinSnap(file, numP, numS):  # read in snapshots from h5-file
    g, q_S, t_S = [], [], []
    with h5.File(file, "r") as f:
        keys = list(f.keys())[:numS]
        for i in range(numP + 2):
            g.append(f["/grid/point/{}".format(i)][()])
        for j, key in enumerate(keys):
            q_S.append([])
            t_S.append(f["/{}/".format(key)].attrs["t"])
            for k in range(numP + 2):
                q_S[j].append(f["/{}/field/{}".format(key, k)][()])
    return q_S, g


def vizSnap(q, g, numP, ivar):  # visualize snapshots
    fig = plt.figure(1, figsize=(6.9, 6.9))
    plt.clf()
    plt.ion()
    ax = fig.gca()
    ax.set_facecolor("k")

    vmin, vmax = 1e10, -1e10
    for i in range(numP + 2):
        vmin = np.min([vmin, np.min(q[i][..., ivar])])
        vmax = np.max([vmax, np.max(q[i][..., ivar])])

    for i in range(numP + 2):
        # j-index adds in the periodic images to the plot
        for j in range(4):
            _ = ax.pcolormesh(
                g[i][..., 0],
                g[i][..., 1] - j * 0.6 * numP,
                q[i][..., ivar],
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                shading="gouraud",
                rasterized=True,
            )
            ax.pcolormesh(
                g[i][..., 0],
                g[i][..., 1] + j * 0.6 * numP,
                q[i][..., ivar],
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                shading="gouraud",
                rasterized=True,
            )
    ax.set_aspect("equal")
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1.0, 1.0])

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    plt.show()


def nextPow(d_act):  # compute the closest (larger) power of 2
    return 2 ** (int(log(d_act) / log(2)) + 1)


def fjlt_Matrices(d, n, k, q):  # compute the FJLT-matrices P,D and scaling s
    d2 = nextPow(d)
    s = np.sqrt(d2 / float(d * k))
    D = npr.randint(0, 2, size=d)  # D matrix

    # m    = npr.binomial(k*d,q)
    m = int(k * d * q)  # P transform
    rng = default_rng()
    indc = rng.choice(k * d, size=m, replace=False)
    r, c = np.unravel_index(indc, (k, d))
    d = npr.normal(loc=0, scale=sqrt(1 / q), size=m)
    P = sparse.csr_matrix((d, (r, c)), shape=(k, d2))
    return P, s, D


def applyFJLT(a, P, s, D):  # apply FJLT to a state vector (efficient version)
    d = len(a)
    k, d2 = P.shape
    D = 2 * D * s - s  # D matrix
    DA = np.zeros(d2)
    DA[:d] = a * D
    HDA = FWHT(DA)  # Walsh-Hadamard transform
    return P.dot(HDA)


def FWHT(x):  # fast Walsh-Hadamard transform
    x = x.squeeze()
    N = x.size
    G, M = N // 2, 2  # number of groups, number of members in each group
    y = np.zeros((N // 2, 2))  # first stage
    y[:, 0], y[:, 1] = x[0::2] + x[1::2], x[0::2] - x[1::2]
    x = y.copy()
    for i in range(2, int(log(N, 2)) + 1):  # second and further stage
        y = np.zeros((G // 2, M * 2))
        y[0 : G // 2, 0 : M * 2 : 4] = x[0:G:2, 0:M:2] + x[1:G:2, 0:M:2]
        y[0 : G // 2, 1 : M * 2 : 4] = x[0:G:2, 0:M:2] - x[1:G:2, 0:M:2]
        y[0 : G // 2, 2 : M * 2 : 4] = x[0:G:2, 1:M:2] - x[1:G:2, 1:M:2]
        y[0 : G // 2, 3 : M * 2 : 4] = x[0:G:2, 1:M:2] + x[1:G:2, 1:M:2]
        x = y.copy()
        G //= 2
        M *= 2
    x = y[0, :]
    x = x.reshape((x.size, 1))
    return x.flatten() / sqrt(float(N))  # introduced sqrt to make FWHT = invFWHT


if __name__ == "__main__":

    numP = 1  # number of passages
    numS = 100  # number of snapshots
    q_S, g = readinSnap(numP, numS)

    s_id = 0  # pick snapshot to visualize
    ivar = 0  # pick variable to visualize (p,s,u,v)
    nx1, ny1, nz1 = np.shape(q_S[0][0])
    nx2, ny2, nz2 = np.shape(q_S[0][1])
    nx3, ny3, nz3 = np.shape(q_S[0][2])
    nn = nx1 * ny1 + nx2 * ny2 + nx3 * ny3

    Q = np.zeros((nn * 4, numS))
    for j in range(numS):
        q1, q2, q3 = q_S[j][0].flatten(), q_S[j][1].flatten(), q_S[j][2].flatten()
        Q[:, j] = np.hstack((q1, q2, q3))

    # vizSnap(q,g,numP,ivar)

    d = nn * 4  # dimensionality
    n = 6  # embedding dimension; snapshots
    e = 0.01  # maximum distortion
    k = int(log(n) / e / e)  # embedding dimension
    q = log(n) * log(n) / d  # sparsity
    print(" parameters: dimensionality =       ", d)
    print(" parameters: snapshots =            ", n)
    print(" parameters: max distortion (pct) = ", e * 100)
    print(" parameters: embedding dim  =       ", k, " (", d * n // k, "times)")
    print(" parameters: sparsity =             ", q)
    print(" ")

    t0 = time.time()
    P, s, D = fjlt_Matrices(d, n, k, q)
    t1 = time.time()
    print("generating matrices took ", t1 - t0, " sec")

    dt = 0
    B = np.zeros((k, numS))
    for j in range(numS):
        t0 = time.time()
        b = applyFJLT(Q[:, j], P, s, D)
        B[:, j] = b
        t1 = time.time()
        dt += t1 - t0
    print("applying matrices took on average ", dt / numS, " sec")

    print(" now processing ", numS, " snapshots ")
    ifa, imm = 0, 20
    for ii in range(imm):
        i, j = np.random.randint(0, numS // 2), np.random.randint(numS // 2 + 1, numS)
        xy = np.linalg.norm(Q[:, i] - Q[:, j])
        XY = np.linalg.norm(B[:, i] - B[:, j])
        dd = abs(xy - XY) / xy
        print(
            "TEST  (",
            i,
            j,
            ") orig = ",
            xy,
            " compr = ",
            XY,
            " distortion percentage = ",
            dd * 100,
        )
        if dd > e:
            ifa += 1

    print("===================")
    print(ifa, " failures in ", imm, " with a failure rate (in percent) of ", ifa / imm)

    # do sparsity-promoting DMD on the FJLT-compressed data (incomplete)
    gammas = [1.0e-3, 2, 10, 20, 30, 50, 100, 1000]
    spdmd_list = [SpDMD(svd_rank=30, gamma=gm, rho=1.0e4).fit(B) for gm in gammas]
    std_dmd = DMD(svd_rank=30).fit(B)

    for dd in spdmd_list:
        print(dd.amplitudes)
