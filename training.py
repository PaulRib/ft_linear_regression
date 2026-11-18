import pandas as pd


def normalizing(value: list[int]) -> list[float]:
    mean = sum(value) / len(value)
    var = sum((v - mean) ** 2 for v in value) / len(value)
    normalized = [(v - mean) / (var ** 0.5) for v in value]
    return normalized


def training(theta: list[float]) -> list[float]:
    min_error = 0.000001
    total_error = [0.0, 0.0]
    data = pd.read_csv("data.csv")
    km = data["km"].tolist()
    price = data["price"].tolist()
    normal_price = normalizing(price)
    normal_km = normalizing(km)
    previous_cost = float("inf")
    for i in range(100):
        estimated_price = [theta[0] + (theta[1] * v) for v in normal_km]
        error_cost = sum((p - v) ** 2 for p, v in zip(estimated_price, normal_price)) / (2 * len(normal_km))
        if abs(previous_cost - error_cost) < min_error:
            break;
        previous_cost = error_cost
        total_error[0] = sum(p - v for p, v in zip(estimated_price, normal_price))
        total_error[1] = sum((p - v) * k for p, v, k in zip(estimated_price, normal_price, normal_km))
        tmp_theta0 = 0.1 * (total_error[0] / len(normal_price))
        tmp_theta1 = 0.1 * (total_error[1] / len(normal_price))
        theta[0] -= tmp_theta0
        theta[1] -= tmp_theta1
        print(f"Iteration {i + 1} : cost : {error_cost}")
    return theta

if __name__ == "__main__":
    training([0.0, 0.0])