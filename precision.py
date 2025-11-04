#!/usr/bin/env python3

import json
from utils import estimate_price, load_model, normalize_data, read_data

def r2_score(y_true, y_pred):
	mean_y = sum(y_true) / len(y_true)
	ss_total = sum([(yi - mean_y) ** 2 for yi in y_true])
	ss_res = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
	return 1 - (ss_res / ss_total) if ss_total != 0 else 0

def precision_verbose(r2):
	if r2 > 0.9:
		return "✅ \033[92mExcellent\033[0m"
	elif r2 > 0.7:
		return "👍 \033[92mGood\033[0m"
	elif r2 > 0.5:
		return "⚠️ \033[93mModerate\033[0m"
	else:
		return "❌ \033[91mPoor\033[0m"

def main():
	print("[0] Loading model and data...")

	model = load_model()
	if model is None:
		return 1

	x, y = read_data("data.csv")

	print("[1] Evaluating model precision...")

	predictions = [estimate_price(xi, model["theta0"], model["theta1"]) for xi in x]

	r2 = r2_score(y, predictions)
	print(f"Model precision (R²): {r2:.4f} {precision_verbose(r2)}")
	return 0

if __name__ == "__main__":
	raise SystemExit(main())