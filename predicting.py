import pandas as pd
import training as tr
import data as dt


def predicting(theta: list[float], mileage: float) -> float:
    n_theta = theta
    data = pd.read_csv("data.csv")
    km = data["km"].tolist()
    price = data["price"].tolist()
    n_price, m_price, s_price = dt.normalizing(price)
    n_km, m_km, s_km = dt.normalizing(km)
    n_theta = tr.training(theta, n_km, n_price)
    theta = dt.denormalize(n_theta, s_km, s_price, m_km, m_price)
    estimatedPrice = theta[0] + (theta[1] * mileage)
    return estimatedPrice


def main():
    try:
        theta = [0.0, 0.0]
        mileage = float(input("Enter the mileage of the car:\n"))
        estimatedPrice = theta[0] + (theta[1] * mileage)
        print(f"Estimated price before training : {estimatedPrice}")
        estimatedPrice = predicting(theta, mileage)
        print(f"Estimated price after training : {estimatedPrice}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
