import matplotlib.pyplot as plt


def plot(km: list[float], price: list[float]):
	plt.scatter(km, price)
	plt.xlabel("KM")
	plt.ylabel("Price")
	plt.grid(True)
	plt.show()


def double_bar(data1: list[float], data2: list[float]):
	fig, axes = plt.subplots(1, 2)

	axes[0].bar(range(len(data1)), data1.sort())
	axes[1].bar(range(len(data2)), data2.sort())

	plt.tight_layout()
	plt.show()