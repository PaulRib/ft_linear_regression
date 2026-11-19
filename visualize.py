import matplotlib.pyplot as plt
import pandas as pd
import json
import data as dt


def plot(km: list[float], price: list[float]):
    fig, axes = plt.subplots(1, 2)
    normal_km = dt.normalizing(km)
    normal_price = dt.normalizing(price)

    axes[0].scatter(km, price)
    axes[0].set_title("Données originales")
    axes[0].set_xlabel("Kilométrage")
    axes[0].set_ylabel("Prix (€)")

    axes[1].scatter(normal_km[0], normal_price[0])
    axes[1].set_title("Données normalisées")
    axes[1].set_xlabel("Kilométrage (Z-score)")
    axes[1].set_ylabel("Prix (Z-score)")

    plt.tight_layout()
    plt.show()


def regressive_line(km: list[float], price: list[float],
                    estimatedPrice: list[float]):
    plt.scatter(km, price, label="Real prices")
    plt.plot(km, estimatedPrice, color="red", label="Linear regression")
    plt.xlabel("KM")
    plt.ylabel("Price")
    plt.title("Linear regression")
    plt.legend()
    plt.grid(True)
    plt.show()


def error_percent(estimatedPrice: list[float], price: list[float]):
    meanPrice = sum(price) / len(price)
    totalError = sum((p - e) ** 2 for p, e in zip(price, estimatedPrice))
    totalVar = sum((p - meanPrice) ** 2 for p in price)
    r2 = 1 - (totalError / totalVar)
    print(f"R² score: {r2:.4f}")


def main():
    try:
        data = pd.read_csv("data.csv")
        km = data["km"].tolist()
        price = data["price"].tolist()
        plot(km, price)
        with open("theta.json", "r") as file:
            theta = json.load(file)
        estimatedPrice = [theta[0] + (theta[1] * k) for k in km]
        regressive_line(km, price, estimatedPrice)
        error_percent(estimatedPrice, price)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
