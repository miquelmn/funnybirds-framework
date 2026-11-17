import torch, math, time

EPSILON = 1e-7

# NMF by multiplictive updates
def NMF(V, k, W=None, H=None, max_iter=200, tol=1e-4, device=None):
    """ Performs Non-negative Matrix Factorization (NMF) to decompose a given matrix into the
    product of two non-negative matrices, W and H.

    This method alternates between updates to W and H using multiplicative rules until
    convergence is achieved, as determined by a tolerance threshold or the maximum
    number of iterations.

    Args:
        V (torch.Tensor): The non-negative input matrix to be factorized. Must have a shape
            of (n_samples, n_features).
        k (int): The number of components (rank) for the factorization.
        W (torch.Tensor, optional): Initial guess for matrix W with shape (n_samples, k).
            Defaults to None, in which case it will be initialized randomly.
        H (torch.Tensor, optional): Initial guess for matrix H with shape (k, n_features).
            Defaults to None, in which case it will be initialized randomly.
        max_iter (int, optional): The maximum number of iterations for the optimization.
            Defaults to 200.
        tol (float, optional): Tolerance for early stopping. If the relative change in
            approximation error falls below this value, the process stops. Defaults to 1e-4.
        device (torch.device, optional): The computational device on which to perform the
            factorization (e.g., "cpu" or "cuda"). Defaults to "cuda" if available, otherwise "cpu".

    Returns:
        tuple: A tuple (W, H), where:
            - W (torch.Tensor): The computed non-negative matrix W with dimensions (n_samples, k).
            - H (torch.Tensor): The computed non-negative matrix H with dimensions (k, n_features).

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scale = math.sqrt(V.mean() / k)

    if W is None:
        W = torch.randn(V.size(0), k).to(device)
        W *= scale

    update_H = True
    if H is None:
        H = torch.randn(k, V.size(1)).to(device)
        H *= scale
    else:
        update_H = False

    W = torch.abs(W)
    H = torch.abs(H)

    error_at_init = approximation_error(V, W, H, square_root=True)
    previous_error = error_at_init

    VH = None
    HH = None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(V, W, H, update_H=update_H, VH = VH, HH = HH)
        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H, square_root=True)

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    return W, H


def multiplicative_update_step(V, W, H, update_H=True, VH=None, HH=None):
    # update operation for W
    if VH is None:
        assert HH is None
        Ht = torch.t(H)
        VH = torch.mm(V, Ht)
        HH = torch.mm(H, Ht)

    WHH = torch.mm(W, HH)
    WHH[WHH == 0] = EPSILON
    W *= VH / WHH

    if update_H:
        # update operation for H (after updating W)
        Wt = torch.t(W)
        WV = torch.mm(Wt, V)
        WWH = torch.mm(torch.mm(Wt, W), H)
        WWH[WWH == 0] = EPSILON
        H *= WV / WWH
        VH, HH = None, None

    return W, H, VH, HH


# NMF objective
def approximation_error(V, W, H, square_root=True):
    # Frobenius norm
    return torch.norm(V - torch.mm(W, H))