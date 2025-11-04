#!/usr/bin/env python3

import matplotlib.pyplot as plt
from utils import load_model, read_data, estimate_price

def plot_data_and_regression():
	print("[0] Loading model and data...")

	# Load trained model
	model = load_model()
	if model is None:
		return 1

	theta0, theta1 = model["theta0"], model["theta1"]

	# Load data
	try:
		x, y = read_data("data.csv")
	except Exception as e:
		print(f"\033[91mError reading data: {e}\033[0m")
		return 1

	x_min, x_max = min(x), max(x)
	x_line = [x_min, x_max]
	y_line = [estimate_price(xi, theta0, theta1) for xi in x_line]

	print("[1] Creating visualization...")

	plt.figure(figsize=(10, 6))

	plt.scatter(x, y, color='blue', label='Data points', zorder=0)
	plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Linear regression\n(y = {theta0:.2f} + {theta1:.6f}x)', zorder=1)

	plt.xlabel('Mileage (km)', fontsize=12)
	plt.ylabel('Price', fontsize=12)

	plt.title('Linear Regression: Car Price vs Mileage', fontsize=14, fontweight='bold')
	plt.grid(True, alpha=0.3, linestyle='--')
	plt.legend(loc='upper right', fontsize=10)

	plt.tight_layout()

	print("\033[92m✅ The plot is now available\033[0m")
	print("\033[93mClose the plot window to exit\033[0m")

	plt.show()

	return 0

def main():
	return plot_data_and_regression()

if __name__ == "__main__":
	raise SystemExit(main())
