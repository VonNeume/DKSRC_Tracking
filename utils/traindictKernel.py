#-*- coding: utf-8 -*-
import torch
import torch.nn.functional as Func
from concurrent import futures

ex = futures.ThreadPoolExecutor()

def getSparseCombinationKernelSingle(A, T, KzX, KXX):
    # Assume the dict is normalized
    zest = torch.zeros([KXX.shape[0], 1]).cuda()
    indices = []
    KzXA = KzX.mm(A)
    KXXA = KXX.mm(A)
    for i in range(T):
        # For all vectors ai, compute the residual products
        tau = KzXA - zest.T.mm(KXXA)
        Imax = torch.argmax(torch.abs(tau))
        indices.append(Imax)
        Ai = A[:, indices[:]]
        tmp = Ai.T.mm(KXX).mm(Ai)
        # tmp = tmp + torch.eye(tmp.shape[0]).cuda()*1e-4     # For inv-able
        xs = torch.linalg.inv(tmp).mm(KzX.mm(Ai).T)
        zest = Ai.mm(xs)

    sparseX = torch.zeros([A.shape[1], 1]).cuda()#.astype(np.float32)
    sparseX[indices, :] = xs
    return sparseX

# @jit(nopython=True, parallel=True)
def getSparseCombinationKernel(n_atoms, z, X, T, A, kfncs):
    gamma = torch.zeros((n_atoms, z.shape[1])).cuda()
    KXX = kfncs(X, X)
    def task(i):
        KzX = kfncs(z[:, i, None], X)  # X is the set of training samples, the sparse code is calculated separately for each corresponding sample
        gamma[:, i, None] = getSparseCombinationKernelSingle(A, T, KzX, KXX)

    for _ in ex.map(task, range(z.shape[1])):
        pass
    return gamma


def evalKKSVD(z, X, A, T0, kfnc, Gamma=None, importance=None):
    Kzz = kfnc(z, z)
    KXX = kfnc(X, X)
    r = torch.zeros([1, z.shape[1]]).cuda()
    # t = time.time()
    # for q in range(z.shape[1]):
    def task(q):
        KzX = kfnc(z[:, q, None], X)
        gamma = getSparseCombinationKernelSingle(A, T0, KzX, KXX)
        if importance is not None:
            importance[gamma!=0] += 1
        r[:, q] = Kzz[q, q] - 2 * KzX.mm(A).mm(gamma) + gamma.T.mm(A.T).mm(KXX).mm(A).mm(gamma)

    for _ in ex.map(task, range(z.shape[1])):
        pass

    if importance is not None:
        return r, importance
    else:
        return r

# Main func for our work
def DKSRC(X, Y, labels, n_atoms, T, iters, kfncs):
    # Initialize the dictionary matrix A for different classes
    rho = 1e-5
    A = []

    for i in range(len(labels)):
        Xi = X[:, Y==labels[i]]
        A.append(Func.normalize(torch.rand([Xi.shape[1], n_atoms[i]]), p=2, dim=0).cuda())

    # Discriminative dictionary learning
    for j in range(len(labels)):
        Xj = X[:, Y == labels[j]]  # [:, :trainlen]
        Xjbar = X[:, Y != labels[j]]  # [:, :trainlen]
        KXXj = kfncs(Xj, Xj)
        KXjXbar = kfncs(Xj, Xjbar)
        KXbarXBarj = kfncs(Xjbar, Xjbar)

        for _ in range(iters):
            # =========== Sparse coding step ==========================
            Gamma = getSparseCombinationKernel(n_atoms[j], Xj, Xj, T, A[j], kfncs)
            GammaBar = getSparseCombinationKernel(n_atoms[j], Xjbar, Xj, T, A[j], kfncs)
            # =========== Dictionary learning step ==========================
            Nj, Njbar = Xj.shape[1], Xjbar.shape[1]
            TmpA, TmpB = Gamma.mm(KXXj).mm(Gamma.T) / Nj, GammaBar.mm(KXbarXBarj).mm(GammaBar.T) / Njbar
            F = TmpA - rho*TmpB
            F2 = F - min(torch.linalg.eigvalsh(F))*torch.eye(F.shape[1]).cuda()
            E = Gamma.mm(KXXj)/Nj - rho*GammaBar.mm(KXjXbar.T) / Njbar
            A[j] = updateA(A[j], E, F2)

        A[j] = Func.normalize(A[j], p=2, dim=0)

    return A

def updateA(A, E, F):
    cost_new = -2*torch.trace(A.mm(E)) + torch.trace(A.mm(F).mm(A.T))
    cost_old = cost_new + 100
    max_iter = 200
    iter = 0
    while torch.abs(cost_new-cost_old)>1e-4 and iter < max_iter:
        cost_old = cost_new
        for j0 in range(A.shape[1]):
            if F[j0,j0] != 0:
                a = 1/F[j0,j0]*(E[j0, :, None]-A.mm(F[:, j0, None]))+A[:, j0, None]
                A[:, j0, None] = a / max(torch.linalg.norm(a, 2), 1)
            else:
                A[:, j0] = A[:, j0] / max(torch.linalg.norm(A[:, j0], 2), 1)

        cost_new = -2 * torch.trace(A.mm(E)) + torch.trace(A.mm(F).mm(A.T))
        iter = iter + 1
    return A