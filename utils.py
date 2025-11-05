import csv, json

MAX_MILEAGE = 1e10
MAX_PRICE = 1e15

def read_data(file_path):
	x, y = [], []
	with open(file_path, mode='r') as file:
		reader = csv.DictReader(file, skipinitialspace=True)

		for i, row in enumerate(reader):
			try:
				km = float(row["km"])
				price = float(row["price"])
			except ValueError as e:
				raise ValueError(f"Invalid data format at row {i}: {e}")
			except OverflowError:
				raise ValueError(f"Numeric value out of range at row {i}")

			if km < 0:
				raise ValueError(f"Invalid km value '{km}' at row {i}. Mileage must be non-negative.")
			elif km > MAX_MILEAGE:
				raise ValueError(f"Invalid km value '{km}' at row {i}. Mileage exceeds maximum limit.")
			if price < 0:
				raise ValueError(f"Invalid price value '{price}' at row {i}. Price must be non-negative.")
			elif price > MAX_PRICE:
				raise ValueError(f"Invalid price value '{price}' at row {i}. Price exceeds maximum limit.")

			x.append(km)
			y.append(price)

	if not x or not y:
		raise ValueError(f"The CSV file '{file_path}' is empty or contains no valid data.")

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