#!/usr/bin/env python3

import json
from utils import estimate_price, load_model

def main():
	model = load_model()
	if model is None:
		return 1
	
	theta0, theta1 = model["theta0"], model["theta1"]
	
	try:
		mileage = float(input("Enter the mileage (in km): "))
	except ValueError:
		print("\033[91mError: Invalid input.\033[0m")
		return 1

	price = estimate_price(mileage, theta0, theta1)

	if price < 0:
		print(f"\033[93mWarning: Predicted price is negative ({price:.2f}). The mileage may be outside the training data range.\033[0m")
		price = 0
	
	print(f"\033[92mEstimated price for {mileage} km: {price:.2f}\033[0m")

	return 0

if __name__ == "__main__":
	raise SystemExit(main())