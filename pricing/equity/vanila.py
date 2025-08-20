import numpy as np
from scipy.stats import norm
from models.FD_solver import FiniteDifferenceEngine
from utils.enum import PDEScheme, MonteCarloMethod
from market.utils import extract_pricing_inputs

# -------------------------------
# 1. PDE Pricing
# -------------------------------

def compute_coefficients(S, dS, sigma, r, dt, scheme):
    if scheme == PDEScheme.EXPLICIT:
        # FTCS scheme: time-explicit, space-central
        a = 0.5 * dt * (sigma ** 2 * S[1:-1] ** 2 / dS ** 2 - r * S[1:-1] / dS)
        b = 1 - dt * (sigma ** 2 * S[1:-1] ** 2 / dS ** 2 + r)
        c = 0.5 * dt * (sigma ** 2 * S[1:-1] ** 2 / dS ** 2 + r * S[1:-1] / dS)
    else:
        # Implicit and Crank-Nicolson use central differencing
        a = 0.5 * sigma**2 * S[1:-1]**2 / dS**2 - 0.5 * r * S[1:-1] / dS
        b = -sigma**2 * S[1:-1]**2 / dS**2 - r
        c = 0.5 * sigma**2 * S[1:-1]**2 / dS**2 + 0.5 * r * S[1:-1] / dS

    # Pad to full grid size
    a = np.concatenate(([0], a, [0]))
    b = np.concatenate(([0], b, [0]))
    c = np.concatenate(([0], c, [0]))

    return a, b, c


def price_vanilla_fd(market_data, valuation_date, expiry_date, strike, option_type: str, scheme: PDEScheme):
    S0, sigma, r, T = extract_pricing_inputs(market_data, valuation_date, expiry_date, strike)
    discount_curve = market_data.discount_curve()

    # Grid parameters
    Smax = max(10.0 * strike, 10.0 * S0)
    M = 700
    dS = Smax / M
    S = np.linspace(0.0, Smax, M + 1)

    # Time steps
    if scheme == PDEScheme.EXPLICIT:
        # CFL-based stability
        dt = 0.9 * dS**2 / (sigma**2 * Smax**2 + 1e-8)
        N = int(T / dt) + 1
        dt = T / N  # recompute to fit exactly
    else:
        N = 400
        dt = T / N

    # Coefficients (scheme-aware)
    a, b, c = compute_coefficients(S, dS, sigma, r, dt, scheme)

    # Payoff function
    payoff_fn = (
        lambda Sarr: np.maximum(Sarr - strike, 0.0)
        if option_type.lower() == "call"
        else np.maximum(strike - Sarr, 0.0)
    )

    # Boundary conditions
    def bc_fn(t, xS):
        tau = max(T - t, 0.0)
        df = discount_curve.discount_factor_T(tau)
        
        if option_type.lower() == "call":
            if xS <= 0.0:
                return 0.0
            elif xS >= Smax:
                return xS - strike * df
        elif option_type.lower() == "put":
            if xS <= 0.0:
                return strike * df
            elif xS >= Smax:
                return 0.0
        return 0.0



    # Solve PDE
    engine = FiniteDifferenceEngine(Smax=Smax, M=M, N=N, method=scheme)
    S_grid, V_grid = engine.solve(a=a, b=b, c=c, T=T, payoff_fn=payoff_fn, boundary_cond_fn=bc_fn)

    # Enforce non-negativity
    V_grid = np.maximum(V_grid, 0.0)

    # Interpolate result
    price = float(np.interp(S0, S_grid, V_grid))

    # Debug output
    print(f"\nPDE {scheme.name} Results:")
    print(f"Grid: M={M}, N={N}, dS={dS:.4f}, dt={dt:.6f}")
    print(f"S0={S0:.2f}, Interpolated Value={price:.4f}")
    print(f"Boundary Values: V(0)={V_grid[0]:.4f}, V(Smax)={V_grid[-1]:.4f}")

    return price




# -------------------------------
# 2. Analytical Black–Scholes Pricing
# -------------------------------



def price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, option_type: str):
    S0, sigma, r, T = extract_pricing_inputs(market_data, valuation_date, expiry_date, strike)
    
    if T <= 0:
        payoff = max(S0 - strike, 0) if option_type.lower() == "call" else max(strike - S0, 0)
        return payoff

    # Get discount factor from curve
    discount_curve = market_data.discount_curve()
    df = discount_curve.discount_factor_T(T)  # Use curve's discounting method
    
    d1 = (np.log(S0 / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S0 * norm.cdf(d1) - strike * df * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = strike * df * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return max(price, 0)


# -------------------------------
# 3. Monte Carlo Pricing
# -------------------------------



def price_vanilla_mc(market_data, valuation_date, expiry_date, strike, option_type, method: MonteCarloMethod, num_paths=2**17):
    S0, sigma, r, T = extract_pricing_inputs(market_data, valuation_date, expiry_date, strike)
    discount_curve = market_data.discount_curve()
    df = discount_curve.discount_factor_T(T)  # Use curve's discounting method
    
    rng = np.random.default_rng(1234)
    z = rng.standard_normal(num_paths)

    if method == MonteCarloMethod.ANTITHETIC:
        z = np.concatenate([z, -z])
    elif method == MonteCarloMethod.QUASI_RANDOM:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=1, scramble=True, seed=1234)
        u = sampler.random(num_paths)
        z = norm.ppf(u).flatten()

    # Path generation - ensure proper drift
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    # Payoff calculation
    if option_type.lower() == "call":
        payoff = np.maximum(ST - strike, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(strike - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Control variate adjustment
    if method == MonteCarloMethod.CONTROL_VARIATE:
        
        control_payoff = (
            np.maximum(ST - strike, 0.0) if option_type.lower() == "call"
            else np.maximum(strike - ST, 0.0)
            )
        control_exact = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, option_type)
        
        cov = np.cov(payoff, control_payoff, ddof=1)[0, 1]
        var_control = np.var(control_payoff, ddof=1)
        beta = cov / var_control
        payoff = payoff - beta * (control_payoff - control_exact)

    # Use curve discounting for final price
    return df * np.mean(payoff)