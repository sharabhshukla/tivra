from typing import Optional
import torch
from tivra.extractor import PyomoExtractor
from tivra.utils.pyomo_generator import create_large_lp
from tivra.device import TivraAccelerator, get_torch_device

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
                 accelerator: Optional[TivraAccelerator] = TivraAccelerator.CPU,
                 data_type: torch.dtype = torch.float64):
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
        self.device, self.data_type = get_torch_device(accelerator)

    def _prox_f(self, v, tau, c, var_lb=None, var_ub=None):
        """
        Prox of f(x) = c^T x is the affine shift: v - tau*c.
        """
        # Affine shift for the objective
        x_tilde = v - tau * c

        # Projection onto variable bounds
        if var_lb is not None:
            x_tilde = torch.maximum(x_tilde, var_lb)
        if var_ub is not None:
            x_tilde = torch.minimum(x_tilde, var_ub)

        return x_tilde

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

    def solve(self, model, *args, **kwargs):
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

        if "max_iter" in kwargs:
            self.max_iter = kwargs["max_iter"]
        extractor = PyomoExtractor(model)
        # A -> Constraint matrix
        # c -> cost coefficient
        # b_lower -> lower bound constraint on constraint
        # b_upper -> upper bounds on constraint
        # lb -> lower bounds on vars
        # ub -> upper bounds on vars
        A, c, b_lower, b_upper, senses, lb, ub = extractor.extract_all()


        # Ensure all inputs are on the desired device
        A = torch.tensor(A, device=self.device, dtype=torch.float32)
        b_min = torch.tensor(b_lower, device=self.device, dtype=torch.float32)
        b_max = torch.tensor(b_upper, device=self.device, dtype=torch.float32)
        c = torch.tensor(c, device=self.device, dtype=torch.float32)
        var_lb = torch.tensor(lb, device=self.device, dtype=torch.float32)
        var_ub = torch.tensor(ub, device=self.device, dtype=torch.float32)

        m, n = A.shape

        # Estimate spectral norm of A (largest singular value)
        # (Alternatively, one could do an iterative power-method approach.)
        # We use torch.linalg.svdvals(A).max() to get the spectral norm.
        L = torch.linalg.norm(A, 2)

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
            x = self._prox_f(x_in, tau, c, var_lb, var_ub)

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
        # transfer torch tensor to cpu before returning
        x = x.detach().cpu().numpy()

        return x


# -------------
# Example usage (Tiny random example)
if __name__ == "__main__":

    # For reproducibility
    torch.manual_seed(0)
    model, exact_solution = create_large_lp(num_vars=100, var_bound=(0,1))

    solver = TivraSolver(max_iter=2000, tol=1e-7, theta=1.0, verbose=True, accelerator=TivraAccelerator.CPU)
    x_solver_sol = solver.solve(model)

    print("\nSolution x =", x_solver_sol)
    print("Exact Solution")
    print(exact_solution)

