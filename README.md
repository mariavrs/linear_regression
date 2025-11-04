# Linear Regression: Car Price Prediction

A Python implementation of **single-feature linear regression** using **gradient descent** to predict car prices based on mileage. This project demonstrates fundamental machine learning concepts without relying on external ML libraries.

### Core Components
- **Training Program** (`train.py`): Implements gradient descent algorithm with normalization 
- **Prediction Program** (`predict.py`): Uses trained parameters to interactively estimate car prices based on mileage
- **Precision Calculator** (`precision.py`): Evaluates model accuracy using R² metric
- **Visualization Tool** (`plot.py`): Plots data distribution and regression line

## 🚀 Quick Start

### Prerequisites

```bash
python3
matplotlib (for visualization only)
```

#### File Requirements
- `data.csv` (initial requirement) - Training dataset (mileage, price pairs)
- `model.json` (created and updated by `train.py`) - Stores trained parameters (θ₀, θ₁)

### Installation

```bash
git clone https://github.com/mariavrs/linear_regression.git
cd ft_linear_regression
```

### Usage

#### 1. Train the Model

```bash
python3 train.py
```

The training process will:
- Load data from `data.csv`
- Normalize features for optimal convergence
- Apply gradient descent with automatic convergence detection
- Save the trained model to `model.json`

#### 2. Make Predictions

```bash
python3 predict.py
```

Interactive prompt for price estimation:
```
Enter the mileage (in km): 100000
Estimated price for 100000.0 km: xxx
```

#### 3. Evaluate Model Precision

```bash
python3 precision.py
```

Outputs the R² score to assess model accuracy

#### 4. Visualize Results

```bash
python3 plot.py
```

Generates a scatter plot with the regression line overlaid on the training data.

## 🧮 Algorithm Details

### Hypothesis Function

The linear regression model uses the following hypothesis to estimate prices:

```
estimatePrice(mileage) = θ₀ + (θ₁ × mileage)
```

**Parameters:**
- `θ₀` (theta0): Intercept - base price when mileage is zero
- `θ₁` (theta1): Slope - price change per kilometer

**Initial State:** Both θ₀ and θ₁ start at 0 before training.

### Gradient Descent Algorithm

The training program implements **batch gradient descent** with simultaneous parameter updates:

```python
tmpθ₀ = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i])
tmpθ₁ = learningRate × (1/m) × Σ((estimatePrice(mileage[i]) - price[i]) × mileage[i])

θ₀ = θ₀ - tmpθ₀
θ₁ = θ₁ - tmpθ₁
```

Where:
- `m` = number of training examples
- `learningRate` = step size for parameter updates (α = 0.1)
- Σ = sum over all training examples (i = 0 to m-1)

### Training Configuration

**Hyperparameters:**
- Learning rate (α): 0.1
- Max iterations: 10,000
- Convergence threshold: 1e-7 (cost change between iterations)

**Cost Function (Mean Squared Error):**
```
J(θ₀, θ₁) = (1/(2m)) × Σ(estimatePrice(mileage[i]) - price[i])²
```

The algorithm stops when cost improvement falls below the threshold or max iterations is reached.

### Data Normalization

Min-max normalization is applied to prevent feature scaling issues:

```
x_normalized = (x - min(x)) / (max(x) - min(x))
```

After training, parameters are denormalized to work with original scale data.

## 📈 Dataset

The `data.csv` file contains 24 samples with two features:
- **km**: Car mileage (input feature)
- **price**: Car price (target variable)

Sample data:
```csv
km,price
240000,3650
139800,3800
150500,4400
...
```

## 📄 License

This project is created for educational purposes.
