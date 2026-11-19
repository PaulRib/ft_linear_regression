# ft_linear_regression

## Introduction

Linear regression is a machine learning method that models the linear
relationship between a dependent variable (usually denoted by $y$) and
one or more independent variables (usually denoted by $x$). It can be
used to predict continuous numerical values.

In **ft_linear_regression**, the goal is to predict a car’s price ($y$)
from its mileage ($x$) using a simple linear regression model trained
with gradient descent.

The model uses the following equation:

$$
\hat{y} = \theta_0 + \theta_1 x
$$

Here, $\hat{y}$ is the estimated price, $\theta_0$ is the intercept, and
$\theta_1$ is the slope. Both parameters are initially set to zero.

## 1. Preparing and normalizing the data

The training dataset (`data.csv`) contains the mileage and price of **24
cars**. Because mileage and price have different numerical scales, both
features are standardized before training. This makes gradient descent
easier to optimize with a single learning rate.

This project uses **Z-score standardization**:

$$
x' = \frac{x - \mu}{\sigma}
$$

Where:

- $x'$ is the standardized value (Z-score).
- $x$ is the original value.
- $\mu$ is the mean of the dataset values.
- $\sigma$ is their standard deviation.

For example, a value two standard deviations above the mean has a
Z-score of $+2$, while a value 1.5 standard deviations below the mean
has a Z-score of $-1.5$.

The means and standard deviations of mileage and price are retained to
convert the trained parameters back to their original units.

## 2. Training with gradient descent

The training algorithm iteratively adjusts $\theta_0$ and $\theta_1$ to
reduce the difference between predicted and actual prices. All equations
in this section use **standardized** mileage and prices.

### Step 1 — Estimate prices

For each car $i$, the model computes:

$$
\hat{y}_i = \theta_0 + \theta_1 x_i
$$

During the first iteration, both parameters are zero, so every estimated
price is zero in standardized units.

### Step 2 — Evaluate the cost

The implementation uses a **squared-error cost function**:

$$
J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=0}^{m-1}(\hat{y}_i-y_i)^2
$$

Where:

- $m$ is the number of cars in the dataset.
- $\hat{y}_i$ is the estimated standardized price of car $i$.
- $y_i$ is its actual standardized price.
- $J$ is the overall cost to minimize.

Squaring each difference prevents positive and negative errors from
canceling each other out. The factor $1/2$ simplifies the gradient
formulas.

The current cost is compared with the previous iteration’s cost.
Training stops when the absolute improvement becomes smaller than
**$10^{-6}$**, or after **100 iterations**, whichever comes first.

### Step 3 — Calculate the gradients

First, the accumulated errors are calculated:

$$
\mathrm{error}_0 = \sum_{i=0}^{m-1}(\hat{y}_i-y_i)
$$

$$
\mathrm{error}_1 = \sum_{i=0}^{m-1}(\hat{y}_i-y_i)x_i
$$

Where:

- $\mathrm{error}_0$ is the sum of the signed differences between predicted and actual standardized prices.
- $\mathrm{error}_1$ is the sum of those differences weighted by the corresponding standardized mileage.
- $\hat{y}_i$ is the predicted standardized price for car $i$.
- $y_i$ is the actual standardized price for car $i$.
- $x_i$ is the standardized mileage for car $i$.
- $i$ is the index of a car in the dataset.
- $m$ is the total number of cars in the dataset.

The first sum measures the signed prediction errors. The second weights
each error by the corresponding standardized mileage, providing the
information needed to adjust the slope.

The parameter adjustments are then calculated using a learning rate
$\alpha = 0.1$:

$$
\mathrm{tmp}\theta_0 = \alpha\frac{\mathrm{error}_0}{m}
$$

$$
\mathrm{tmp}\theta_1 = \alpha\frac{\mathrm{error}_1}{m}
$$

Where:

- $\mathrm{tmp}\theta_0$ is the adjustment calculated for the intercept.
- $\mathrm{tmp}\theta_1$ is the adjustment calculated for the slope.
- $\alpha$ is the learning rate, set to $0.1$ in this project.
- $\mathrm{error}_0$ is the accumulated signed prediction error.
- $\mathrm{error}_1$ is the accumulated prediction error weighted by standardized mileage.
- $m$ is the total number of cars in the dataset.

### Step 4 — Update both parameters

The two adjustments are calculated **before either parameter is
changed**, so they are based on the same iteration’s predictions:

$$
\theta_0 \leftarrow \theta_0 - \mathrm{tmp}\theta_0
$$

$$
\theta_1 \leftarrow \theta_1 - \mathrm{tmp}\theta_1
$$

These steps repeat until the stopping condition is met.

## 3. Converting the parameters back to real units

Because the model was trained on standardized data, its parameters must
be converted before predicting prices directly from mileage in
kilometers.

The real-unit slope is:

$$
\theta_{1,\mathrm{real}} = \theta_{1,\mathrm{norm}}\frac{\sigma_{\mathrm{price}}}{\sigma_{\mathrm{km}}}
$$

The real-unit intercept is:

$$
\theta_{0,\mathrm{real}} = \mu_{\mathrm{price}} + \theta_{0,\mathrm{norm}}\sigma_{\mathrm{price}} - \theta_{1,\mathrm{real}}\mu_{\mathrm{km}}
$$

Where:

- $\theta_{1,\mathrm{real}}$ is the slope converted to euros per kilometer.
- $\theta_{1,\mathrm{norm}}$ is the slope learned from standardized data.
- $\theta_{0,\mathrm{real}}$ is the intercept converted to euros.
- $\theta_{0,\mathrm{norm}}$ is the intercept learned from standardized data.
- $\sigma_{\mathrm{km}}$ is the standard deviation of the original mileages.
- $\sigma_{\mathrm{price}}$ is the standard deviation of the original prices.
- $\mu_{\mathrm{km}}$ is the mean of the original mileages.
- $\mu_{\mathrm{price}}$ is the mean of the original prices.

This conversion is performed once after training. The two real-unit
parameters are then saved to `theta.json` for use by the prediction
program.

## 4. Predicting a car’s price

The prediction program prompts the user for a mileage and applies the
original linear equation using the saved parameters:

$$
\mathrm{estimatedPrice} = \theta_{0,\mathrm{real}} + \theta_{1,\mathrm{real}}\times\mathrm{mileage}
$$

Unlike the training program, prediction does not need to read or
normalize the dataset. Before training, the parameters default to zero.

## 5. Project files and usage

| File            | Purpose                                                                                |
|-----------------|----------------------------------------------------------------------------------------|
| `training.py`   | Loads the dataset, trains the model, converts the parameters, and writes `theta.json`. |
| `predicting.py` | Reads the saved parameters and predicts a price for the entered mileage.               |
| `data.py`       | Provides Z-score standardization and parameter conversion functions.                   |
| `visualize.py`  | Displays the data, the fitted regression line, and the model’s $R^2$ score.            |
| `data.csv`      | Training data containing mileage and price.                                            |
| `theta.json`    | Parameters saved by the training program.                                              |

Install the Python dependencies used by the project:

``` bash
python3 -m pip install pandas matplotlib
```

Train the model:

``` bash
python3 training.py
```

Then estimate a price:

``` bash
python3 predicting.py
```

To display the visualizations and model score:

``` bash
python3 visualize.py
```

Run these commands from the project directory so that the programs can
find `data.csv` and `theta.json`.

## 6. Bonus: visualization and model evaluation

The visualization program provides a scatter plot of car prices against
mileage and a second plot showing the fitted regression line alongside
the original observations.

It also calculates the **coefficient of determination**, $R^2$:

$$
R^2 = 1 - \frac{\sum_{i=0}^{m-1}(y_i-\hat{y}_i)^2}{\sum_{i=0}^{m-1}(y_i-\bar{y})^2}
$$

Where:

- $R^2$ is the coefficient of determination.
- $y_i$ is the actual price of car $i$.
- $\hat{y}_i$ is the predicted price of car $i$.
- $\bar{y}$ is the mean of all actual car prices.
- $i$ is the index of a car in the dataset.
- $m$ is the total number of cars in the dataset.
- $\sum_{i=0}^{m-1}(y_i-\hat{y}_i)^2$ is the sum of squared prediction errors.
- $\sum_{i=0}^{m-1}(y_i-\bar{y})^2$ is the total sum of squares around the mean price.

The score measures how well
the fitted line explains the variation in the dataset’s prices. A value
of $1$ indicates a perfect fit; a score can be negative if the model
performs worse than predicting the mean price for every car.

For the supplied dataset and saved parameters, the model produces an
**$R^2$ score of approximately 0.7330**. This evaluation is performed on
the training data, so it does not measure performance on unseen cars.
