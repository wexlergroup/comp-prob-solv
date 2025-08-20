---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Chapter 12: The Monte Carlo Method

## Learning Objectives

By the end of this lecture, you should be able to

1. Understand the basic principles of the Monte Carlo method.
2. Apply Monte Carlo sampling to estimate integrals.
3. Explain the concept of importance sampling and its benefits.

## Introduction to Monte Carlo Method

The **Monte Carlo method** is a powerful computational technique used to solve problems by relying on random sampling. To illustrate its basic idea, let’s start with a simple example: evaluating the integral of $ f(x) = x^2 $ over the interval $[0, 1]$

$$
I = \int_0^1 x^2 \, dx
$$

### Relating Integrals to Averages

Recall that the average value of a function $ f(x) $ over an interval $[a, b]$ is given by

$$
\langle f(x) \rangle = \frac{1}{b - a} \int_a^b f(x) \, dx
$$

In this case, we can write the integral as the average of $ x^2 $ over $[0, 1]$

$$
I = \langle x^2 \rangle = \int_0^1 x^2 \, dx
$$

### Monte Carlo Estimation

Instead of solving this integral analytically, we can approximate it by randomly sampling points from the interval $[0, 1]$ and evaluating $ x^2 $ at those points. This is the essence of the **Monte Carlo method**.

Suppose we generate $ N $ random numbers $ x_i $ uniformly distributed between 0 and 1. We can approximate the integral as the average of $ x_i^2 $

$$
I \approx \frac{1}{N} \sum_{i=1}^N x_i^2
$$

This gives us an estimate for the integral. The larger the number of random samples $ N $, the closer the estimate gets to the true value.

### Python Implementation

The following Python code demonstrates how to estimate the integral using Monte Carlo sampling.

```{code-cell} ipython3
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of random samples
N = 1000

# Generate N random numbers uniformly distributed between 0 and 1
x = np.random.rand(N)

# Estimate the average of x^2 over [0, 1]
average_x_squared = np.mean(x**2)

# Estimate the variance of the estimator
variance = np.var(x**2)

# Print the result with a descriptive message
print(f"The estimated average of x^2 over [0, 1] with {N} samples is: {average_x_squared:.6f}")
print(f"The estimated variance of the estimator is: {variance:.6f}")
print(f"The exact value of the integral is: {1/3:.6f}")
```

```{tip}
Increasing the number of random samples $ N $ reduces the variance of the estimate and improves the accuracy of the approximation.
```

## Importance Sampling

````{margin}
```{note}
The **uniform distribution** is the simplest probability distribution, characterized by a constant probability density function over a given interval.
```
````

In the previous example, we used a uniform distribution to generate random numbers for our Monte Carlo simulation. However, we can often improve the efficiency of these simulations by employing a different probability distribution—this approach is known as **importance sampling**.

### Motivation for Importance Sampling

Let's reconsider the goal of estimating the integral

$$
I = \int_0^1 x^2 \, dx
$$

Using a uniform distribution, the Monte Carlo estimator would directly sample from this integral. However, efficiency can be improved by selecting a distribution that matches the behavior of the integrand, *i.e.*, a distribution that emphasizes regions where $x^2$ is larger. This results in a smaller variance of the estimator.

### Rewriting the Integral

We can rewrite the integral as

$$
I = \int_0^1 \frac{x^2}{g(x)} g(x) \, dx
$$

where $g(x)$ is any probability distribution (with support over [0, 1]) that can be used to sample points more efficiently. The idea is to choose a $g(x)$ that is similar to the shape of the function $x^2$. This way, the sampling is concentrated in regions where the function $x^2$ contributes more to the integral.

### Choosing a Suitable $g(x)$

A good candidate for $g(x)$ is the **Beta distribution** with parameters $\alpha = 3$ and $\beta = 1$. The probability density function of the Beta distribution is

$$
g(x) = \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}
$$

where $B(\alpha, \beta)$ is the Beta function, and in this case, it biases the samples toward larger values of $x$, aligning well with the shape of $x^2$.

### Monte Carlo Estimator with Importance Sampling

Using the Beta distribution, the integral can be approximated as

$$
I \approx \frac{1}{N} \sum_{i=1}^N \frac{x_i^2}{g(x_i)}
$$

where $x_i$ are samples drawn from the Beta distribution.

### Implementation in Python

The following code demonstrates how to estimate the integral of $x^2$ using importance sampling.

```{code-cell} ipython3
import numpy as np
from scipy.stats import beta

# Set random seed for reproducibility
np.random.seed(42)

# Number of random samples
N = 100

# Parameters of the Beta distribution (biasing towards larger x)
alpha, beta_param = 3, 1

# Generate N random samples from the Beta distribution
x = np.random.beta(alpha, beta_param, size=N)

# Evaluate the integrand at the random numbers (x^2)
f_x = x**2

# Evaluate the importance sampling weights
g_x = beta.pdf(x, alpha, beta_param)  # PDF of the Beta distribution

# Since the target is the uniform distribution on [0, 1], its PDF is constant 1
# So the weights become 1/g(x)
weights = 1 / g_x

# Estimate the integral using importance sampling
estimated_integral = np.mean(f_x * weights)

# Estimate the variance of the estimator
variance = np.var(f_x * weights)

# Print the result with a descriptive message
print(f"The estimated value of the integral of x^2 over [0, 1] using importance sampling is: {estimated_integral:.6f}")
print(f"The estimated variance of the estimator is: {variance}")
print(f"The exact value of the integral is: {1/3:.6f}")
```
