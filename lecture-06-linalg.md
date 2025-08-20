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

# Chapter 6: Balancing Chemical Equations and Systems of Linear Algebraic Equations

## Learning Objectives

By the end of this lecture, you should be able to:

1. **Balance chemical equations** using algebraic methods and matrix representation.
2. **Solve systems of linear equations** for stoichiometric coefficients using Python and NumPy.
3. **Apply the null space method** to find balanced coefficients for chemical reactions.
4. **Interpret and generalize solutions** to balance hydrocarbon combustion reactions and other chemical equations.

## Balancing Chemical Equations

In the chemical sciences, one of the foundational skills is **balancing chemical equations** to ensure that no mass is lost or created, as dictated by the law of conservation of mass. This process ensures that the number of atoms of each element is the same on both sides of the reaction.

Consider the combustion of methane in oxygen:

$$
a \text{CH}_4(g) + b \text{O}_2(g) \rightarrow c \text{CO}_2(g) + d \text{H}_2\text{O}(g)
$$

The goal is to determine the stoichiometric coefficients $a$, $b$, $c$, and $d$ that balance the equation. We can do this systematically by ensuring the number of each type of atom is equal on both sides of the reaction. A useful approach is to create a table that tracks the number of atoms in the reactants and products for each element:

| Element | Reactants       | Products         |
|---------|-----------------|------------------|
| C       | $a$             | $c$              |
| H       | $4a$            | $2d$             |
| O       | $2b$            | $2c + d$         |

From this table, we can see the relationships between the variables. First, we know that for carbon to balance, $a = c$, since there is one carbon atom in both methane and carbon dioxide. Substituting this into the table gives:

| Element | Reactants       | Products         |
|---------|-----------------|------------------|
| C       | $a$             | $a$              |
| H       | $4a$            | $2d$             |
| O       | $2b$            | $2a + d$         |

Next, we balance hydrogen. Since there are 4 hydrogen atoms in methane, and each water molecule contains 2 hydrogen atoms, we know that $4a = 2d$, which simplifies to $d = 2a$. Updating the table:

| Element | Reactants       | Products         |
|---------|-----------------|------------------|
| C       | $a$             | $a$              |
| H       | $4a$            | $4a$             |
| O       | $2b$            | $2a + 2a = 4a$   |

Finally, we balance oxygen. We have $2b$ oxygen atoms on the reactant side and $4a$ on the product side, so $2b = 4a$, meaning $b = 2a$.

Now, substituting these values into the chemical equation, with $a = 1$ for simplicity, we get:

$$
\text{CH}_4(g) + 2 \text{O}_2(g) \rightarrow \text{CO}_2(g) + 2 \text{H}_2\text{O}(g)
$$

Thus, the balanced chemical equation for the combustion of methane is:

$$
\text{CH}_4(g) + 2 \text{O}_2(g) \rightarrow \text{CO}_2(g) + 2 \text{H}_2\text{O}(g)
$$

This systematic approach shows that we can think of balancing chemical equations as solving a set of linear equations for the stoichiometric coefficients.

## Systems of Linear Algebraic Equations

Balancing chemical equations can be thought of as solving a system of linear algebraic equations. If we have the same number of equations as unknowns, we can solve for the unknowns using linear algebra techniques. Let’s revisit the methane combustion problem and express it as a system of equations.

For the balanced reaction:

$$
a \text{CH}_4(g) + b \text{O}_2(g) \rightarrow c \text{CO}_2(g) + d \text{H}_2\text{O}(g)
$$

We can write the following set of equations based on the element balances:

1. **Carbon balance**: $a - c = 0$ (since there is one carbon atom in both methane and carbon dioxide)
2. **Hydrogen balance**: $4a - 2d = 0$ (4 hydrogen atoms in methane, 2 per water molecule)
3. **Oxygen balance**: $2b - 2c - d = 0$ (2 oxygen atoms in each oxygen molecule, balancing with oxygen in carbon dioxide and water)

This system of equations can be expressed as:

$$
\begin{align*}
a - c &= 0 \\
4a - 2d &= 0 \\
2b - 2c - d &= 0 \\
\end{align*}
$$

Next, we can write this in **matrix form** as:

$$
\begin{align*}
\begin{bmatrix}
1 & 0 & -1 & 0 \\
4 & 0 & 0 & -2 \\
0 & 2 & -2 & -1 \\
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
c \\
d \\
\end{bmatrix}
&=
\begin{bmatrix}
0 \\
0 \\
0 \\
\end{bmatrix}
\end{align*}
$$

Here, the matrix on the left represents the coefficients of the unknowns $a$, $b$, $c$, and $d$, and the vector on the right is the zero vector because we are balancing the equation.

More concisely, this can be written as:

$$
\mathbf{A} \mathbf{x} = \mathbf{0}
$$

Where:

- $\mathbf{A}$ is the matrix of coefficients
- $\mathbf{x}$ is the vector of unknowns $[a, b, c, d]$
- $\mathbf{0}$ is the zero vector.

## Solving the System of Equations

Now, let's apply what we’ve learned to solve the combustion of methane using Python. We’ll employ the `numpy` library for matrix operations and the `scipy.linalg.null_space` function to solve the system by finding the null space of the coefficient matrix.

### Step 1: Import the Necessary Libraries

We'll first import the libraries needed for our computation.

```{code-cell} ipython3
import numpy as np
from scipy.linalg import null_space
```

### Step 2: Define the Coefficient Matrix, $\mathbf{A}$

Next, we define the matrix $\mathbf{A}$, which represents the coefficients of the unknowns $a$, $b$, $c$, and $d$ in the system of equations:

```{code-cell} ipython3
A = np.array([[1, 0, -1, 0],
              [4, 0, 0, -2],
              [0, 2, -2, -1]])
```

This matrix corresponds to the carbon, hydrogen, and oxygen balances, respectively, for the combustion reaction.

### Step 3: Compute the Null Space

To solve for the stoichiometric coefficients, we find the null space of matrix $\mathbf{A}$. This will provide the ratios between $a$, $b$, $c$, and $d$ that satisfy the system of linear equations:

```{code-cell} ipython3
# Calculate the null space of matrix A
null_vec = null_space(A)
```

The result is a vector of relative coefficients. However, we need to convert these into the smallest possible integer values to balance the equation correctly.

### Step 4: Normalize and Convert to Integer Coefficients

We can normalize the null space vector by dividing by its smallest positive value and then round to obtain integer coefficients:

```{code-cell} ipython3
# Normalize the coefficients to the smallest integer values
coefficients = null_vec[:, 0] / np.min(null_vec[null_vec > 0])
coefficients = np.round(coefficients).astype(int)

coefficients
```

The resulting coefficients correspond to the stoichiometric coefficients for the reaction:

- $a = 1$
- $b = 2$
- $c = 1$
- $d = 2$

### Step 5: The Balanced Chemical Equation

The balanced chemical equation for the combustion of methane is:

$$
\text{CH}_4(g) + 2\text{O}_2(g) \rightarrow \text{CO}_2(g) + 2\text{H}_2\text{O}(g)
$$

This method can be generalized to balance any chemical equation using matrix algebra and Python.

---

```{admonition} General Case for Hydrocarbon Combustion
:class: tip
For the combustion of any alkane, the general form for balancing the equation is:

$$
\text{C}_n\text{H}_{2n+2}(g) + (3n+1)\text{O}_2(g) \rightarrow n\text{CO}_2(g) + (n+1)\text{H}_2\text{O}(g)
$$

More generally, for any hydrocarbon (saturated or unsaturated), the formula for the stoichiometric coefficients is:

$$
\text{C}_x\text{H}_y + \left(x + \frac{y}{4}\right)\text{O}_2 \rightarrow x\text{CO}_2 + \frac{y}{2}\text{H}_2\text{O}
$$
```

---

### Summary

In this section, we demonstrated how to solve the combustion reaction of methane by representing it as a system of linear equations. By using Python, we computed the null space of the coefficient matrix and found the stoichiometric coefficients needed to balance the equation. This same approach can be extended to balance more complex chemical reactions.

## Example: Reduction of Tin(IV) Oxide by Hydrogen

Next, let’s explore the reduction of tin(IV) oxide by hydrogen to produce tin and water. The unbalanced chemical equation for this reaction is:

$$
\text{SnO}_2(s) + \text{H}_2(g) \rightarrow \text{Sn}(s) + \text{H}_2\text{O}(g)
$$

### Balancing the Equation by Hand

Before we jump into solving this problem using Python, take a moment to try balancing this equation manually. Start by creating a table that tracks the number of atoms of each element on both sides of the reaction.

| Element | Reactants | Products |
|---------|-----------|----------|
| Sn      | 1         | 1        |
| O       | 2         | 1        |
| H       | 2         | 2        |

From this, you can see that the oxygen atoms are not balanced. We have 2 oxygen atoms on the left (from SnO₂) and only 1 oxygen atom on the right (in H₂O). Therefore, we’ll need to add a coefficient of 2 to the water molecule on the product side.

$$
\text{SnO}_2(s) + \text{H}_2(g) \rightarrow \text{Sn}(s) + 2\text{H}_2\text{O}(g)
$$

Now, check the hydrogen atoms. We now have 4 hydrogen atoms on the right side (from 2 H₂O molecules), so we need 2 H₂ molecules on the left side to balance the hydrogen.

The balanced equation is:

$$
\text{SnO}_2(s) + 2\text{H}_2(g) \rightarrow \text{Sn}(s) + 2\text{H}_2\text{O}(g)
$$

```{admonition} Take a Moment
:class: warning
Did you try balancing the equation by hand? Understanding the manual process will help you see how Python can automate and streamline this process.
```

### Solving the Equation Using Python

Once you’ve manually balanced the equation, let’s confirm the result using Python. By writing the system of equations for the element balances, we can solve for the stoichiometric coefficients using matrix algebra.

```{code-cell} ipython3
import numpy as np
from scipy.linalg import null_space

# Define the matrix of coefficients
A = np.array([[1, 0, -1, 0],   # Sn balance
              [2, 0, 0, -1],   # O balance
              [0, 2, 0, -2]])  # H balance

# Compute the null space of the matrix A
null_vec = null_space(A)

# Normalize and convert to integer coefficients
coefficients = null_vec[:, 0]
coefficients = coefficients / np.min(coefficients[coefficients > 0])
coefficients = np.round(coefficients).astype(int)

# Output the coefficients
coefficients
```

This gives us the same result: $1$ SnO₂ reacts with $2$ H₂ to produce $1$ Sn and $2$ H₂O.

---

### Recap

In this example, we manually balanced the reduction of tin(IV) oxide by hydrogen and confirmed our result using Python. You can apply this process to more complex reactions where balancing by hand might be more challenging.

By leveraging matrix algebra and Python, we can efficiently balance chemical equations, saving time and ensuring accuracy, especially for reactions involving multiple reactants and products.

<!-- ## Example: Reduction of Tin(IV) Oxide by Hydrogen

Let's consider the reduction of tin(IV) oxide by hydrogen to form tin and water.

$$
\begin{align*}
\text{SnO}_2(s) + \text{H}_2(g) &\rightarrow \text{Sn}(s) + \text{H}_2\text{O}(g) \\
\end{align*}
$$

```{admonition} Wait a Minute!
:class: warning
Can you try to do this by hand before you do it in Python?
``` -->
