def normalizing(value: list[int]) -> tuple[list[float], float, float]:
    mean = sum(value) / len(value)
    var = sum((v - mean) ** 2 for v in value) / len(value)
    normalized = [(v - mean) / (var ** 0.5) for v in value]
    return normalized, mean, var ** 0.5


def denormalize(n_theta: list[float], s_km: float,
                s_price: float, m_km: float, m_price: float) -> list[float]:
    r_theta = [0.0, 0.0]
    r_theta[1] = n_theta[1] * (s_price / s_km)
    r_theta[0] = m_price + n_theta[0] * s_price - r_theta[1] * m_km
    return r_theta
