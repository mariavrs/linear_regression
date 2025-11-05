#!/usr/bin/env python3

import json
from utils import denormalize_thetas, normalize_data, read_data

LEARNING_RATE = 0.1
MAX_ITERATIONS = 10000
CONVERGENCE_THRESHOLD = 1e-7

def train_model(x, y):
	m = len(x)
	theta0, theta1 = 0.0, 0.0
	prev_cost = float('inf')

	for iteration in range(MAX_ITERATIONS):
		# Compute predictions (θ₀ + θ₁ * x) and errors in one pass
		errors = [theta0 + theta1 * xi - yi for xi, yi in zip(x, y)]
		
		# Calculate cost (MSE)
		cost = (1/(2*m)) * sum(e**2 for e in errors)

		# Check convergence
		cost_change = abs(prev_cost - cost)
		if cost_change < CONVERGENCE_THRESHOLD:
			print(f"\033[92mConverged at iteration {iteration} (final cost: {cost:.9f})\033[0m")
			break
		prev_cost = cost

		# Compute gradients
		tmp_theta0 = LEARNING_RATE * (1/m) * sum(errors)
		tmp_theta1 = LEARNING_RATE * (1/m) * sum(e * xi for e, xi in zip(errors, x))

		# Update θ₀, θ₁
		theta0 -= tmp_theta0
		theta1 -= tmp_theta1

		# Print progress
		if iteration % 1000 == 0:
			print(f"Iteration {iteration:>6}: cost={cost:.9f}, θ₀={theta0:.6f}, θ₁={theta1:.6f}")

	# Print final values if didn't converge
	if iteration == MAX_ITERATIONS - 1:
		print(f"\033[93mReached max iterations without full convergence (final cost: {cost:.9f})\033[0m")

	return theta0, theta1

def main():
	print("[0] Preparing data...")

	try:
		x, y = read_data("data.csv")
	except Exception as e:
		print(f"\033[91mError reading data: {e}\033[0m")
		return 1

	x, min_x, max_x = normalize_data(x)
	y, min_y, max_y = normalize_data(y)

	print("[1] Training model with normalized data...")

	theta0_norm, theta1_norm = train_model(x, y)
	theta0, theta1 = denormalize_thetas(theta0_norm, theta1_norm, min_x, max_x, min_y, max_y)

	print("\033[92mTraining completed\033[0m")
	print("[2] Saving model...")

	try:
		with open("model.json", "w", encoding="utf-8") as f:
			json.dump({"theta0": theta0, "theta1": theta1}, f)
		print("\033[92mModel saved to model.json\033[0m")
	except Exception as e:
		print(f"\033[91mError saving model: {e}\033[0m")
		return 1

	return 0

if __name__ == "__main__":
	raise SystemExit(main())