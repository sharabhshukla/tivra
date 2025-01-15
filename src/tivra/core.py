from enum import Enum
import torch


class TivraAccelerator(str, Enum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    XPU = "xpu"
    HPU = "hpu"


class TivraSolver:
    """
    A simple Primal-Dual Hybrid Gradient (Chambolle–Pock) solver
    for the LP: min c^T x  subject to  b_min <= A x <= b_max,
    rewritten to use PyTorch tensors.
    """

    def __init__(self,
                 max_iter=5000,
                 tol=1e-6,
                 theta=1.0,
                 verbose=False,
                 logging_interval: int = 50,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize PDHG solver parameters.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default=5000).
        tol : float, optional
            Tolerance for stopping criterion (default=1e-6).
        theta : float, optional
            Extrapolation parameter in [0,1] (default=1.0).
        verbose : bool, optional
            If True, print progress information (default=False).
        device : str or torch.device, optional
            Device to place tensors on, e.g. 'cpu' or 'cuda'. If None, defaults to CPU.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.theta = theta
        self.verbose = verbose
        self.logging_interval = logging_interval
        self.device = device if device is not None else torch.device('cpu')

    def _prox_f(self, v, tau, c):
        """
        Prox of f(x) = c^T x is the affine shift: v - tau*c.
        """
        return v - tau * c

    def _prox_g_star(self, v, sigma, b_min, b_max):
        """
        Prox of g^*(y) for g(z) = i_{b_min <= z <= b_max}.

        Closed-form update (per coordinate):
          if v_i > sigma*b_max[i]:  u_i = v_i - sigma*b_max[i]
          if v_i < sigma*b_min[i]:  u_i = v_i - sigma*b_min[i]
          else:                     u_i = 0
        """
        u = torch.zeros_like(v)

        idx_up = (v > sigma * b_max)
        u[idx_up] = v[idx_up] - sigma * b_max[idx_up]

        idx_dn = (v < sigma * b_min)
        u[idx_dn] = v[idx_dn] - sigma * b_min[idx_dn]

        return u

    def _extrapolate(self, x_k, x_k_minus_1):
        """
        Extrapolation step: overline{x}^k = x^k + theta * (x^k - x^{k-1}).
        """
        return x_k + self.theta * (x_k - x_k_minus_1)

    def solve(self, A, b_min, b_max, c):
        """
        Solve the LP:  min c^T x  subject to  b_min <= A x <= b_max
        using the PDHG method.

        Parameters
        ----------
        A : torch.Tensor, shape (m, n)
            Constraint matrix.
        b_min : torch.Tensor, shape (m,)
            Lower bounds (finite).
        b_max : torch.Tensor, shape (m,)
            Upper bounds (finite).
        c : torch.Tensor, shape (n,)
            Objective vector.

        Returns
        -------
        x : torch.Tensor, shape (n,)
            Approximate minimizer of the LP.
        """

        # Ensure all inputs are on the desired device
        A = A.to(self.device)
        b_min = b_min.to(self.device)
        b_max = b_max.to(self.device)
        c = c.to(self.device)

        m, n = A.shape

        # Estimate spectral norm of A (largest singular value)
        # (Alternatively, one could do an iterative power-method approach.)
        # We use torch.linalg.svdvals(A).max() to get the spectral norm.
        L = torch.linalg.svdvals(A).max()

        tau = 1.0 / L
        sigma = 1.0 / L

        # Initialize primal and dual variables
        x = torch.zeros(n, device=self.device, dtype=A.dtype)
        x_old = torch.zeros_like(x)
        y = torch.zeros(m, device=self.device, dtype=A.dtype)

        for k in range(self.max_iter):
            # 1) Extrapolate: overline{x}^k
            x_bar = self._extrapolate(x, x_old)

            # 2) Dual update: y <- prox_{sigma*g^*}( y + sigma * A * x_bar )
            y_in = y + sigma * torch.matmul(A, x_bar)
            y_new = self._prox_g_star(y_in, sigma, b_min, b_max)

            # 3) Primal update: x <- prox_{tau*f}( x - tau * A^T * y_new )
            x_old.copy_(x)  # keep a copy for the next extrapolation
            x_in = x - tau * torch.matmul(A.t(), y_new)
            x = self._prox_f(x_in, tau, c)

            # 4) Check a simple stopping criterion
            norm_diff = torch.norm(x - x_old)
            denom = 1.0 + torch.norm(x)
            if (norm_diff / denom) < self.tol:
                if self.verbose:
                    print(f"[PDHG] Converged at iteration {k+1}")
                break

            y = y_new  # update dual

            # Optional progress info
            if self.verbose and (k+1) % self.logging_interval == 0:
                obj_val = torch.dot(c, x).item()
                print(f"Iter {k+1}:  obj={obj_val:.6g}, step_diff={norm_diff:.3e}")

        return x


# -------------
# Example usage (Tiny random example)
if __name__ == "__main__":

    # For reproducibility
    torch.manual_seed(0)

    # Create random problem data on CPU
    m, n = 5, 3
    A_ = torch.randn(m, n, dtype=torch.double)
    b_min_ = -0.5 * torch.ones(m, dtype=torch.double)
    b_max_ = +1.0 * torch.ones(m, dtype=torch.double)
    c_ = torch.tensor([1.0, 2.0, -0.5], dtype=torch.double)

    solver = TivraSolver(max_iter=2000, tol=1e-7, theta=1.0, verbose=True, device='cpu')
    x_sol = solver.solve(A_, b_min_, b_max_, c_)

    print("\nSolution x =", x_sol)
    print("Objective c^T x =", torch.dot(c_, x_sol).item())
    Ax_ = torch.matmul(A_, x_sol)
    print("Check constraints b_min <= A x <= b_max ?")
    print("  A x =", Ax_.cpu().numpy())
    print("  b_min =", b_min_.cpu().numpy())
    print("  b_max =", b_max_.cpu().numpy())
