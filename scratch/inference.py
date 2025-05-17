from typing import Tuple
import math
from .probability import normal_cdf, inverse_normal_cdf


def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    mu = n * p
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


normal_probability_below = normal_cdf


def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    return 1 - normal_cdf(lo, mu, sigma)


def normal_probability_between(
    lo: float, hi: float, mu: float = 0, sigma: float = 1
) -> float:
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


def normal_probability_outside(
    lo: float, hi: float, mu: float = 0, sigma: float = 1
) -> float:
    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_sided_bounds(
    probability: float, mu: float = 0, sigma: float = 1
) -> float:
    tail_probability = (1 - probability) / 2

    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound


def two_sidded_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)


def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma


def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)

    return (p_B - p_A) / math.sqrt(sigma_A**2 + sigma_B**2)


def B(alpha: float, beta: float) -> float:
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)


def beta_pdf(x: float, alpha: float, beta: float) -> float:

    if x <= 0 or x >= 1:
        return 0

    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
