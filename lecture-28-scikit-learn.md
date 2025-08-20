---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: '1.16.4'
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Chapter 26: `scikit-learn`

## Learning Objectives

By the end of this lecture, you will be able to:

- Use `scikit-learn` to perform supervised learning
- Understand the difference between classification and regression
- Train and evaluate classification models
- Train and evaluate regression models

## `scikit-learn`

`scikit-learn` is a Python package that provides simple and efficient tools for data analysis. It is built on `numpy`, `scipy`, and `matplotlib`. It is open source and commercially usable under the BSD license. It is a great tool for machine learning in Python.

### Installation

To install `scikit-learn`, you can follow the instructions on the [official website](https://scikit-learn.org/stable/install.html). You can install it using `pip`:

```bash
pip install -U scikit-learn
```

## Supervised Learning

In supervised learning, we have a dataset consisting of both input features and output labels. The goal is to learn a mapping from the input to the output. We have two types of supervised learning:

1. Classification: The output is a category.
2. Regression: The output is a continuous value.

### Classification

In classification, we have a dataset consisting of input features and output labels. The goal is to learn a mapping from the input features to the output labels. We can use the `scikit-learn` library to perform classification.

#### Machine Learning by Example: Wine Classification

Let's consider an example of wine classification. We have a dataset of wines with different features such as alcohol content, acidity, etc. We want to classify the wines into different categories based on these features.

##### Step 1: Get the Data

First, we need to load the dataset. We can use the `load_wine` function from `sklearn.datasets` to load the wine dataset.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()
```

```{admonition} Wine Recognition Dataset
:class: tip
The wine recognition dataset is a classic dataset for classification. It contains 178 samples of wine with 13 features each. The features are the chemical composition of the wines, and the target is the class of the wine (0, 1, or 2). You can find more information about the dataset [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset).
```

```{admonition} Your Data
:class: tip
If `pandas` can read your data, you can swap out the `load_wine` function with `pd.read_csv` or any other method you prefer to load your data.
```

##### Step 2: Explore and Visualize the Data

Next, we need to explore and visualize the data to understand its structure and characteristics. We can use `pandas` to explore the data and `seaborn` to visualize it.

```{code-cell} ipython3
df.describe()
```

```{code-cell} ipython3
df['target'].value_counts()
```

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='target')

plt.show()
```

##### Step 3: Preprocess the Data

Before training the model, we need to preprocess the data. This involves splitting the data into input features and output labels, normalizing the input features, and splitting the data into training and testing sets.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```{admonition} Why Split the Data?
:class: tip
Splitting the data into training and testing sets allows us to train the model on one set and evaluate it on another set. This helps us assess the model's performance on unseen data. You can also use [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) to evaluate the model's performance more robustly.
```

```{admonition} Why Scale or "Standardize" the Data?
:class: tip
Standardizing the data (*e.g.*, using `StandardScaler`) ensures that each feature has a mean of 0 and a standard deviation of 1. This can help improve the performance of some machine learning algorithms, especially those that are sensitive to the scale of the input features.
```

##### Step 4: Train a Model

Now that we have preprocessed the data, we can train a classification model. We will use the `LogisticRegression` and `RandomForestClassifier` models from `scikit-learn`.

```{admonition} Logistic Regression
:class: tip
Logistic regression is a linear model used for binary classification. It models the probability of the output being in a particular category. You can find more information about logistic regression [here](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

![Alt text](https://upload.wikimedia.org/wikipedia/commons/c/cb/Exam_pass_logistic_curve.svg)

Canley, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons
```

```{admonition} Random Forest
:class: tip
Random forests are an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes (*i.e.*, the most frequent class) as the prediction. You can find more information about random forests [here](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees).

[DataCamp](https://www.datacamp.com/tutorial/random-forests-classifier-python)
```
<!-- ![Alt text](https://media.datacamp.com/legacy/v1718113325/image_7f309c633f.png) -->

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Train the Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Train the Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Plot the confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=ax[0])
ax[0].set_title('Logistic Regression')
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax[1])
ax[1].set_title('Random Forest')

plt.show()
```

```{admonition} Confusion Matrix
:class: tip
A confusion matrix is a table that is often used to describe the performance of a classification model. It shows the number of true positives, true negatives, false positives, and false negatives. You can find more information about confusion matrices [here](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix).
```

##### Step 5: Evaluate the Model

Finally, we need to evaluate the model's performance. We can use metrics such as accuracy, precision, recall, and F1 score to evaluate the model.

```{code-cell} ipython3
from sklearn.metrics import classification_report

print('Logistic Regression:')
print(classification_report(y_test, y_pred_lr))

print('Random Forest:')
print(classification_report(y_test, y_pred_rf))
```

```{admonition} Classification Report
:class: tip
A classification report shows the precision, recall, F1 score, and support for each class in the classification model. Precision is the ratio of true positives to the sum of true positives and false positives

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

Recall is the ratio of true positives to the sum of true positives and false negatives

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

The F1 score is the harmonic mean of precision and recall

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Support is the number of occurrences of each class in the dataset.
```

##### Step 6: Plot and Interpret the Coefficients

For the logistic regression model, we can plot and interpret the coefficients to understand the importance of each feature in the classification.

```{code-cell} ipython3
import numpy as np

# Ensure feature names are a NumPy array
feature_names = np.array(data.feature_names)

# Sort the coefficients
sorted_idx = lr.coef_[0].argsort()

# Plot the coefficients
plt.figure(figsize=(12, 6))
plt.barh(feature_names[sorted_idx], lr.coef_[0][sorted_idx])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Logistic Regression Coefficients')
plt.show()
```

The plot above shows the coefficients of the logistic regression model. The features with the largest coefficients (in absolute value) are the most important for the classification. The sign of the coefficient indicates the direction of the relationship between the feature and the target. The two features with the largest coefficients are `proline` and `alcalinity_of_ash`.

`proline` is the amount of proline in the wine. Proline is an amino acid that is found in high concentrations in red wines. The coefficient for `proline` is positive, indicating that wines with higher proline content are more likely to be classified as class 2.

```{figure} https://upload.wikimedia.org/wikipedia/commons/4/45/L-Proline.svg
---
width: 200px
name: proline
---
The chemical structure of proline. By Qohelet12, CC0, via Wikimedia Commons
```

`alcalinity_of_ash` is the amount of ash in the wine. Ash is the inorganic residue remaining after the water and organic matter have been removed by heating. The coefficient for `alcalinity_of_ash` is negative, indicating that wines with lower ash content are more likely to be classified as class 2.

```{admonition} Your Data
:class: tip
You can swap out the wine dataset with your own dataset to perform classification on your data. Make sure to preprocess the data, train the model, and evaluate the model as shown in the example above.
```

### Regression

In regression, we have a dataset consisting of input features and continuous output values. The goal is to learn a mapping from the input features to the output values. We can use the `scikit-learn` library to perform regression.

#### Machine Learning by Example: Oxygen Vacancy Formation Energy Prediction

Let's consider an example of regression for predicting the oxygen vacancy formation energy in materials. We have an [Excel file](https://wustl.instructure.com/files/8915454/download?download_frd=1) containing the features of the materials and the oxygen vacancy formation energy. We want to train a regression model to predict the oxygen vacancy formation energy based on the features of the materials.

##### Step 1: Use `pip` or `conda` to Install `openpyxl`

Before we can read the Excel file, we need to install the `openpyxl` library. You can install it using `pip`:

```bash
pip install openpyxl
```

##### Step 2: Get the Data

First, we need to load the dataset. We can use the `pd.read_excel` function from `pandas` to load the Excel file.

```{code-cell} ipython3
df = pd.read_excel('ovfe-deml.xlsx')
df.head()
```

```{admonition} Oxygen Vacancy Formation Energy Dataset
:class: tip
The oxygen vacancy formation energy dataset contains the features of materials and the oxygen vacancy formation energy. The features include the crystal structure (`xtal_str`), the composition (`comp`), the standard enthalpy of formation (`dHf`), the measured and calculated band gaps (`Eg_exp`, `Eg_GW`, and `Eg_DFTU`), the valence band maximum (`O2p_min_VBM`), the difference between the electronegativity of the cation and anion (`dEN`), and the energy above the convex hull (`Ehull_MP`). The energy above the convex hull is a measure of the thermodynamic stability of the material. The higher the energy above the convex hull, the less stable the material. The target is the oxygen vacancy formation energy (`OVFE_calc`). You can find more information about the dataset [here](https://doi.org/10.1021/acs.jpclett.5b00710).
```

```{admonition} Your Data
:class: tip
If `pandas` can read your data, you can swap out the `pd.read_excel` function with `pd.read_csv` or any other method you prefer to load your data.
```

##### Step 3: Explore and Visualize the Data

Next, we need to explore and visualize the data to understand its structure and characteristics. We can use `pandas` to explore the data and `seaborn` to visualize it.

###### Missing Values

Before exploring the data, we need to check for missing values and handle them if necessary.

```{code-cell} ipython3
df.isnull().sum()
```

The measured band gap (`Eg_exp`) and the energy above the convex hull (`Ehull_MP`) have nine and two missing values, respectively. We can drop these columns or impute the missing values with the mean, median, or mode of the column. Let's drop the columns for now.

```{code-cell} ipython3
df.drop(['Eg_exp', 'Ehull_MP'], axis=1, inplace=True)
df.head()
```

```{admonition} Missing Values
:class: tip
Missing values can affect the performance of machine learning models. It is important to handle missing values appropriately by imputing them or dropping the corresponding rows or columns. You can find more information about handling missing values [here](https://scikit-learn.org/stable/modules/impute.html).
```

###### Data Exploration

Now, let's explore the data to understand its structure and characteristics.

```{code-cell} ipython3
df.describe()
```

```{code-cell} ipython3
sns.pairplot(df, kind='reg', diag_kind='kde')
plt.show()
```

##### Step 4: Preprocess the Data

Before training the model, we need to preprocess the data. This involves splitting the data into input features and output labels, normalizing the input features, and splitting the data into training and testing sets.

```{code-cell} ipython3
X = df.drop(['xtal_str', 'comp', 'OVFE_calc', 'OVFE_reg_GW', 'OVFE_reg_DFTU'], axis=1)
y = df['OVFE_calc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

##### Step 5: Train a Model

Now that we have preprocessed the data, we can train a regression model. We will use the `RidgeCV` and `Perceptron` models from `scikit-learn`.

```{admonition} Ridge regression
:class: tip
Ridge regression is a linear model used for regression. It is similar to ordinary least squares regression, but it adds a penalty term to the loss function to prevent overfitting by shrinking the coefficients. You can find more information about ridge regression [here](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression).
```

````{admonition} Multi-layer Perceptron (MLP)
:class: tip
The multi-layer perceptron (MLP) is a type of artificial neural network that consists of multiple layers of nodes. It is a powerful model that can learn complex patterns in the data. You can find more information about MLPs [here](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron).

```{figure} https://scikit-learn.org/stable/_images/multilayerperceptron_network.png
---
width: 400px
name: mlp
---
A multi-layer perceptron (MLP) neural network. [Source](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron)
```
````

```{code-cell} ipython3
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

# Train the Ridge regression model
ridge = RidgeCV()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Train the MLPRegressor model
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Plot the predicted vs. actual values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_ridge, label='Ridge')
plt.scatter(y_test, y_pred_mlp, label='MLP')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Actual Oxygen Vacancy Formation Energy (eV)')
plt.ylabel('Predicted Oxygen Vacancy Formation Energy (eV)')
plt.legend()
plt.show()
```

##### Step 6: Evaluate the Model

Finally, we need to evaluate the model's performance. We can use metrics such as mean squared error (MSE), mean absolute error (MAE), and $R^2$ score to evaluate the model.

```{code-cell} ipython3
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Ridge Regression:')
print('MSE:', mean_squared_error(y_test, y_pred_ridge))
print('MAE:', mean_absolute_error(y_test, y_pred_ridge))
print('R^2:', r2_score(y_test, y_pred_ridge))

print('MLPRegressor:')
print('MSE:', mean_squared_error(y_test, y_pred_mlp))
print('MAE:', mean_absolute_error(y_test, y_pred_mlp))
print('R^2:', r2_score(y_test, y_pred_mlp))
```

```{admonition} Regression Metrics
:class: tip
Mean squared error (MSE) is the average of the squared differences between the predicted and actual values

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Mean absolute error (MAE) is the average of the absolute differences between the predicted and actual values

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

The $R^2$ score is the coefficient of determination and represents the proportion of the variance in the dependent variable that is predictable from the independent variables

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
```

```{admonition} Your Data
:class: tip
You can swap out the oxygen vacancy formation energy dataset with your own dataset to perform regression on your data. Make sure to preprocess the data, train the model, and evaluate the model as shown in the example above.
```

##### Step 7: Plot and Interpret the Coefficients

For the Ridge regression model, we can plot and interpret the coefficients to understand the importance of each feature in the regression.

```{code-cell} ipython3
# Ensure feature names are a NumPy array
feature_names = np.array(X.columns)

# Sort the coefficients
sorted_idx = ridge.coef_.argsort()

# Plot the coefficients
plt.figure(figsize=(12, 6))
plt.barh(feature_names[sorted_idx], ridge.coef_[sorted_idx])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Ridge Regression Coefficients')
plt.show()
```

The plot above shows the coefficients of the Ridge regression model. The features with the largest coefficients (in absolute value) are the most important for the regression. The sign of the coefficient indicates the direction of the relationship between the feature and the target. The feature with the largest coefficient is `dHf`.

```{admonition} Interpretation
:class: tip
`dHf` is the standard enthalpy of formation of the material. The formation reaction of a metal oxide (MO$_x$) can be represented as

$$
\text{M} + \frac{x}{2} \text{O}_2 \rightarrow \text{MO}_x
$$

This reaction can be thought of as an oxidation reaction, where the metal is oxidized to form the metal oxide, and its standard enthalpy change can be thought of as an enthalpy of oxidation. Since oxygen vacancy formation is a reduction reaction, the oxygen vacancy formation energy is inversely related to the standard enthalpy of formation. The coefficient for `dHf` is negative, indicating that materials with lower standard enthalpies of formation have higher oxygen vacancy formation energies.
```

```{admonition} Materials Design
:class: tip
This relationship is exciting because it suggests that we can predict the oxygen vacancy formation energy of metal oxides, which is challenging to measure experimentally, based on their standard enthalpies of formation, which are readily available from databases like the [Materials Project](https://next-gen.materialsproject.org/), [AFLOW](http://aflow.org/), and [OQMD](http://oqmd.org/). This prediction can help guide the design of materials with desired properties for applications like solid oxide fuel cells, oxygen separation membranes, catalysts, and [thermochemical water and carbon dioxide splitting](https://wexlergroup.github.io/research/#solar-thermochemical-hydrogen-production).
```

### Summary

In this lecture, we learned how to use `scikit-learn` to perform supervised learning. We covered classification and regression and trained models on the wine recognition dataset and the oxygen vacancy formation energy dataset. We explored the data, preprocessed it, trained the models, evaluated the models, and interpreted the results. We used logistic regression and random forests for classification and ridge regression and MLPRegressor for regression. We also visualized the data, plotted the confusion matrix, and interpreted the coefficients.
