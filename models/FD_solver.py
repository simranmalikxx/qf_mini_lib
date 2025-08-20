import numpy as np
from utils.enum import PDEScheme 

class FiniteDifferenceEngine:
    """
    Generic finite difference PDE solver.
    Takes precomputed a, b, c coefficients and solves backwards in time.
    """

    def __init__(self, Smax, M, N, method: PDEScheme):
        self.Smax = Smax
        self.M = M
        self.N = N
        self.method = method

    def solve(self, a, b, c, T, payoff_fn, boundary_cond_fn):
        """
        Solve PDE for option value grid.

        Parameters:
        a, b, c : PDE coefficient arrays (size M+1)
        T : maturity
        payoff_fn(S) : payoff at maturity
        boundary_cond_fn(t, S) : boundary value at time t for spot S
        """
        dS = self.Smax / self.M
        dt = T / self.N

        # Stock grid
        S = np.linspace(0, self.Smax, self.M + 1)

        # Initial condition (payoff at maturity)
        V = payoff_fn(S)

        # theta based on method
        if self.method == PDEScheme.EXPLICIT:
            theta = 0.0
        elif self.method == PDEScheme.IMPLICIT:
            theta = 1.0
        elif self.method == PDEScheme.CRANK_NICOLSON:
            theta = 0.5
        else:
            raise ValueError("Unknown finite difference method")

        # For explicit method, check stability condition (optional but recommended)
        if self.method == PDEScheme.EXPLICIT:
            max_b = np.max(np.abs(b))
            if dt > dS**2 / (2 * max_b):
                print(f"Warning: dt = {dt} may be too large for stability. Consider increasing N.")

        # Main loop - backward in time
        for n in range(self.N):
            t = T - n * dt  # Current time
            t_prev = t - dt if n > 0 else t  # Handle first iteration
            
            # Create matrices
            size = self.M - 1
            diag = 1 - theta * dt * b[1:self.M]
            lower = -theta * dt * a[2:self.M]
            upper = -theta * dt * c[1:self.M-1]
            
            M1 = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
            
            diag2 = 1 + (1 - theta) * dt * b[1:self.M]
            lower2 = (1 - theta) * dt * a[2:self.M]
            upper2 = (1 - theta) * dt * c[1:self.M-1]
            
            M2 = np.diag(diag2) + np.diag(lower2, -1) + np.diag(upper2, 1)

            # Right-hand side vector
            rhs = M2 @ V[1:self.M]
            
            # Boundary conditions
            if theta > 0:  # For implicit and CN
                rhs[0] += theta * dt * a[1] * boundary_cond_fn(t, 0)
            if theta < 1:  # For explicit and CN
                rhs[0] += (1 - theta) * dt * a[1] * boundary_cond_fn(t_prev, 0)
            
            if theta > 0:
                rhs[-1] += theta * dt * c[self.M-1] * boundary_cond_fn(t, self.Smax)
            if theta < 1:
                rhs[-1] += (1 - theta) * dt * c[self.M-1] * boundary_cond_fn(t_prev, self.Smax)

            # Solve linear system
            V[1:self.M] = np.linalg.solve(M1, rhs)
            
            # Apply boundary conditions
            V[0] = boundary_cond_fn(t, 0)
            V[-1] = boundary_cond_fn(t, self.Smax)

        return S, V