import pandas as pd


def normalizing(value: list[int]) -> list[float]:
    mean = sum(value) / len(value)
    var = sum((v - mean) ** 2 for v in value) / len(value)
    normalized = [(v - mean) / (var ** 0.5) for v in value]
    return normalized


def training(theta: list[float]) -> list[float]:
    data = pd.read_csv("data.csv")
    km = data["km"].tolist()
    price = data["price"].tolist()
    normal_price = normalizing(price)
    normal_km = normalizing(km)
    estimated_price = [theta[0] + (theta[1] * v) for v in normal_km]

    total_error = sum(p - v for p, v in zip(estimated_price, normal_price))
    tmp_theta0 = 0.1 * (total_error / len(normal_price))
    theta[0] -= tmp_theta0
    total_error = sum((p - v) * k for p, v, k in zip(estimated_price, normal_price, normal_km))
    tmp_theta1 = 0.1 * (total_error / len(normal_price))
    theta[1] -= tmp_theta1
    return theta
