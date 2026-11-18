import numpy as np
import pandas as pd
import training as tr


def predicting():
    theta = [0.0, 0.0]
    mileage = float(input("Enter the mileage of the car:\n"))
    estimatedPrice = theta[0] + (theta[1] * mileage)
    print(f"Estimated price before training : {estimatedPrice}")
    theta = tr.training(theta)
    estimatedPrice = theta[0] + (theta[1] * mileage)
    print(f"Estimated price after 1 iteration : {estimatedPrice}")


def main():
    try:
        predicting()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()