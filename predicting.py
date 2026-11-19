import json


def predicting(theta: list[float], mileage: float) -> float:
    return theta[0] + (theta[1] * mileage)


def main():
    try:
        mileage = int(input("Enter a mileage:\n"))
        theta = [0.0, 0.0]
        estimatedPrice = predicting(theta, mileage)
        print(f"Estimated price for this mileage: \
{estimatedPrice}")
        try:
            with open("theta.json", "r") as file:
                theta = json.load(file)
        except FileNotFoundError as e:
            print(f"Error: {e}. Model can't predict without training!")
            theta = [0.0, 0.0]
        estimatedPrice = predicting(theta, mileage)
        print(f"Estimated price for this mileage after training: \
{int(estimatedPrice)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
