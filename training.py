import json
import pandas as pd
import data as dt


def training(theta: list[float], n_km: list[float],
             n_price: list[float]) -> list[float]:
    min_error = 0.000001
    total_error = [0.0, 0.0]
    previous_cost = float("inf")
    for i in range(100):
        estimated_price = [theta[0] + (theta[1] * v) for v in n_km]
        error_cost = sum((p - v) ** 2 for p, v in
                         zip(estimated_price, n_price)) / (2 * len(n_km))
        if abs(previous_cost - error_cost) < min_error:
            break
        previous_cost = error_cost
        total_error[0] = sum(p - v for p, v in zip(estimated_price, n_price))
        total_error[1] = sum((p - v) * k for p, v, k in zip(estimated_price,
                                                            n_price, n_km))
        tmp_theta0 = 0.1 * (total_error[0] / len(n_price))
        tmp_theta1 = 0.1 * (total_error[1] / len(n_price))
        theta[0] -= tmp_theta0
        theta[1] -= tmp_theta1
        print(f"Iteration {i + 1} : cost : {error_cost}")
    return theta


def main():
    try:
        theta = [0.0, 0.0]
        data = pd.read_csv("data.csv")
        km = data["km"].tolist()
        price = data["price"].tolist()
        if not km or not price:
            raise ValueError("Dataset cannot be empty!")
        n_price, m_price, s_price = dt.normalizing(price)
        n_km, m_km, s_km = dt.normalizing(km)
        n_theta = training(theta, n_km, n_price)
        theta = dt.denormalize(n_theta, s_km, s_price, m_km, m_price)
        with open("theta.json", "w") as file:
            json.dump(theta, file)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
