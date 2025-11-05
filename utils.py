import csv, json

def read_data(file_path):
	x, y = [], []
	with open(file_path, mode='r') as file:
		reader = csv.DictReader(file, skipinitialspace=True)

		for row in reader:
			x.append(float(row["km"]))
			y.append(float(row["price"]))

	if not x or not y:
		raise ValueError(f"Error: The CSV file '{file_path}' is empty or contains no valid data.")

	return x, y

def estimate_price(km, theta0, theta1):
	return theta0 + theta1 * km

def normalize_data(data):
	min_val = min(data)
	max_val = max(data)
	return [(val - min_val) / (max_val - min_val) for val in data], min_val, max_val

def denormalize_thetas(theta0_norm, theta1_norm, min_x, max_x, min_y, max_y):
	theta1 = theta1_norm * (max_y - min_y) / (max_x - min_x)
	theta0 = min_y + (theta0_norm * (max_y - min_y)) - (theta1 * min_x)
	return theta0, theta1

def load_model(path="model.json"):
	try:
		with open(path, "r", encoding="utf-8") as f:
			model = json.load(f)
	except Exception as e:
		print(f"\033[91mError reading {path}:\033[0m {e}\n\033[93mHint: Try to (re)train the model by running 'train.py'.\033[0m")
		return None

	if "theta0" not in model or "theta1" not in model:
		print(f"\033[91mError: theta0 and/or theta1 missing in {path}.\n\033[93mHint: Try to (re)train the model by running 'train.py'.\033[0m")
		return None

	try:
		theta0 = float(model["theta0"])
		theta1 = float(model["theta1"])
	except (ValueError, TypeError) as e:
		print(f"\033[91mError: Invalid theta values in {path}.\n\033[93mHint: Try to (re)train the model by running 'train.py'.\033[0m")
		return None
	
	return {"theta0": theta0, "theta1": theta1}