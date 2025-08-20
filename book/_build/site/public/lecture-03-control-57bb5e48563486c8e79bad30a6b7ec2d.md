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

# Chapter 3: Control Structures in Python

## Introduction

In this lecture, we will explore one of the most crucial aspects of programming: control structures. Control structures are fundamental building blocks in Python, allowing you to control the flow of execution in your programs. By using these structures, you can make your code more dynamic, flexible, and responsive to different conditions and inputs.

### What Are Control Structures?

Control structures dictate the order in which individual statements, instructions, or function calls are executed or evaluated. They enable you to:

- Execute specific blocks of code based on certain conditions.
- Repeat actions or processes efficiently.
- Organize code into reusable sections, making your programs more modular and easier to maintain.

### Key Control Structures in Python

In Python, there are three main types of control structures that we will cover in this lecture:

1. **Conditional Statements:** These allow you to execute code only when certain conditions are met. This is essential for making decisions in your programs.

2. **Loops:** Loops enable you to repeat a block of code multiple times, which is useful for tasks that require iteration, such as processing data sets or performing repetitive calculations.

3. **Functions:** Functions allow you to encapsulate code into reusable blocks. This not only makes your code more organized but also facilitates code reuse and reduces redundancy.

## Learning Objectives

By the end of this lecture, you will be able to:

- Understand and apply conditional statements to control the flow of execution in Python.
- Utilize loops to efficiently repeat tasks and process collections of data.
- Define and use functions to create reusable blocks of code.

---

## Section 1: Conditional Statements

Conditional statements are essential in programming as they allow you to control the flow of your code based on specific conditions. In Python, the most common conditional statements are:

1. **`if` statement**
2. **`if-else` statement**
3. **`if-elif-else` statement**

These structures enable your programs to make decisions and respond accordingly, making your code more dynamic and flexible.

### 1.1 The `if` Statement

The `if` statement is the simplest form of a conditional statement. It allows you to execute a block of code only if a specific condition is true. The syntax is straightforward:

```python
if condition:
    # block of code
```

- **`condition`:** This is an expression that evaluates to either `True` or `False`.
- If the `condition` is `True`, the block of code within the `if` statement is executed.
- If the `condition` is `False`, the code block is skipped.

**Example:**

```{code-cell} ipython3
x = 10

if x > 5:
    print("x is greater than 5")
```

### 1.2 The `if-else` Statement

The `if-else` statement expands on the `if` statement by providing an alternative block of code to execute if the condition is false. This allows you to handle both possibilities:

```python
if condition:
    # block of code for True condition
else:
    # block of code for False condition
```

- If the `condition` is `True`, the code inside the `if` block is executed.
- If the `condition` is `False`, the code inside the `else` block is executed.

**Example:**

```{code-cell} ipython3
x = 1

if x > 5:
    print("x is greater than 5")
else:
    print("x is less than or equal to 5")
```

### 1.3 The `if-elif-else` Statement

The `if-elif-else` statement is a more complex conditional structure that allows you to check multiple conditions sequentially. This is useful when you need to execute different blocks of code based on different conditions:

```python
if condition1:
    # block of code for condition1
elif condition2:
    # block of code for condition2
else:
    # block of code if no conditions are True
```

- The `if` block is executed if `condition1` is `True`.
- If `condition1` is `False` but `condition2` is `True`, the `elif` block is executed.
- If none of the conditions are `True`, the `else` block is executed.

**Example:**

```{code-cell} ipython3
x = 5

if x > 5:
    print("x is greater than 5")
elif x < 5:
    print("x is less than 5")
else:
    print("x is equal to 5")
```

---

## Section 2: Loops

Loops are a fundamental concept in programming, allowing you to execute a block of code multiple times, which is especially useful when working with large datasets, repetitive tasks, or iterative processes. In Python, the two most common types of loops are:

1. **`for` loop**
2. **`while` loop**

### 2.1 The `for` Loop

The `for` loop is used to iterate over a sequence of elements, such as a list, tuple, string, or other iterable objects, and execute a block of code for each element. This makes it incredibly versatile for processing collections of data.

**Syntax:**

```python
for element in sequence:
    # block of code
```

- **`element`:** A variable that takes on the value of each element in the `sequence` one by one.
- **`sequence`:** Any iterable object (e.g., list, tuple, string, range).

**Example:**

```{code-cell} ipython3
for i in range(5):
    print(i)
```

In this example, `range(5)` generates a sequence of numbers from `0` to `4`. The `for` loop iterates over this sequence, printing each number.

#### Looping Through a List

Lists are one of the most common data structures in Python. You can easily loop through a list using a `for` loop:

```{code-cell} ipython3
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(fruit)
```

Here, the `for` loop iterates over the list `fruits`, printing each fruit in the list.

#### Looping Through a String

Strings are also iterable, allowing you to loop through each character individually:

```{code-cell} ipython3
for char in "hello":
    print(char)
```

This loop prints each character in the string `"hello"` on a new line.

#### Looping Through a Dictionary

````{margin}
```{admonition} Lists vs. Dictionaries
- **Lists** (`[]`): Ordered collections of items. Access elements by their position (index), e.g., `my_list[0]`.
- **Dictionaries** (`{}`): Unordered collections of key-value pairs. Access elements by their key, e.g., `my_dict['key']`. Keys must be unique.
```
````

Dictionaries are collections of key-value pairs, and you can loop through both the keys and values:

```{code-cell} ipython3
person = {"name": "Alice", "age": 30, "city": "New York"}

for key, value in person.items():
    print(key, value)
```

In this example, the `for` loop iterates over each key-value pair in the dictionary `person`, printing both the key and its corresponding value.

#### Looping Through a NumPy Array

NumPy arrays are powerful structures for numerical computations. You can loop through them just like any other iterable:

```{code-cell} ipython3
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

for element in arr:
    print(element)
```

This example shows how to iterate over each element in a NumPy array.

#### Looping Through a Pandas DataFrame

Pandas DataFrames are used for handling tabular data. You can loop through each row of a DataFrame using `iterrows()`:

```{code-cell} ipython3
import pandas as pd

data = {"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]}
df = pd.DataFrame(data)

for index, row in df.iterrows():
    print(index, row["name"], row["age"])
```

Here, the loop iterates over each row in the DataFrame `df`, printing the index, name, and age for each row.

```{admonition} Note
:class: note
When we use syntax like `df.iterrows()`, we're calling a **method** of the DataFrame object `df`. A method is like a function but is tied to an object (in this case, the DataFrame). The dot notation (`df.method_name()`) is used to call methods that perform specific actions on the object. So, `iterrows()` is a method of a DataFrame that returns an iterator over its rows, allowing you to loop through each row.
```

#### List Comprehensions

List comprehensions provide a concise way to create lists by applying an expression to each element in an existing iterable. This is an elegant and Pythonic way to create new lists from existing data.

**Syntax:**

```python
new_list = [expression for element in old_list]
```

**Example:**

```{code-cell} ipython3
squares = [x**2 for x in range(5)]
print(squares)
```

In this example, the list comprehension `[x**2 for x in range(5)]` creates a new list object, which is then assigned to the variable `squares`, containing the squares of each element in the range from `0` to `4`.

### 2.2 The `while` Loop

````{margin}
```{admonition} Infinite Loops
An infinite loop is a loop that never terminates. It can occur when the loop condition is always `True`, causing the loop to run indefinitely. To avoid infinite loops, ensure that the loop condition eventually becomes `False`.
```
````

The `while` loop is used to execute a block of code as long as a specified condition is true. This type of loop is particularly useful when the number of iterations is not predetermined, but depends on a condition.

**Syntax:**

```python
while condition:
    # block of code
```

- **`condition`:** An expression that evaluates to either `True` or `False`.
- The code block inside the `while` loop continues to execute as long as the `condition` is `True`.

**Example:**

```{code-cell} ipython3
i = 0

while i < 5:
    print(i)
    i += 1
```

In this example, the variable `i` is initialized to `0`. The `while` loop executes the block of code repeatedly, printing the value of `i` and then incrementing `i` by `1` on each iteration, until `i` reaches `5`.

```{admonition} Exercise
:class: tip
1. **What happens if you move `i = 0` inside the `while` loop block?**  
   Try moving the line `i = 0` inside the `while` loop. What do you expect will happen, and why?
   
2. **What happens if you forget to include `i += 1` inside the loop?**  
   What do you think will occur if you remove the line `i += 1` from the loop? Explain your reasoning and try running the code to see what happens.
```

---

## Section 3: Functions

Functions are a core component of Python programming, allowing you to encapsulate a block of code that performs a specific task. This modular approach helps make your code more organized, reusable, and easier to maintain. Functions in Python are defined using the `def` keyword, followed by the function name and a set of parentheses.

### 3.1 Defining Functions

The basic syntax for defining a function is as follows:

```python
def function_name(parameters):
    # block of code
    return value
```

- **`function_name`:** This is the name you give to your function, which should be descriptive of what the function does.
- **`parameters`:** These are the inputs to the function. You can include multiple parameters, or none at all.
- **`return value`:** The `return` statement is optional and is used to send back a value from the function to the caller. If you don't use `return`, the function will return `None` by default.

**Example:**

```{code-cell} ipython3
def add(x, y):
    return x + y

result = add(3, 5)
print(result)
```

In this example, we define a function `add` that takes two parameters, `x` and `y`, and returns their sum. When we call `add(3, 5)`, the function returns `8`, which is then printed.

### 3.2 Functions with Default Parameter Values

Functions can also have default values for their parameters, making some arguments optional when the function is called.

**Example:**

```{code-cell} ipython3
def greet(name="Alice"):
    return "Hello, " + name

message = greet()
print(message)
```

In this example, the function `greet` has a default parameter `name="Alice"`. When the function is called without providing an argument for `name`, it defaults to `"Alice"`, and the message "Hello, Alice" is printed.

### 3.3 Lambda Functions

Lambda functions are small, anonymous functions that are defined using the `lambda` keyword. Unlike regular functions defined with `def`, lambda functions can have any number of arguments but only one expression. They are often used for short, simple operations that are not reused elsewhere in your code.

**Syntax:**

```python
lambda arguments: expression
```

**Example:**

```{code-cell} ipython3
add = lambda x, y: x + y

result = add(3, 5)
print(result)
```

Here, we define a lambda function `add` that takes two arguments, `x` and `y`, and returns their sum. The lambda function behaves like a regular function but is written in a more compact form.

**Note:** `add` refers to the function itself, while `add(3, 5)` refers to the result of calling the function with the arguments `3` and `5`. In this case, `add(3, 5)` evaluates to `8`.

### 3.4 Using Lambda Functions with Higher-Order Functions

Lambda functions are frequently used as arguments to higher-order functions like `map()`, `filter()`, and `reduce()` because of their concise syntax.

**Syntax of `map()`:**  
`map(function, iterable)`

Here, `function` is the operation to apply, and `iterable` is the collection of items that the function will be applied to.

**Example with `map()`:**

```{code-cell} ipython3
numbers = [1, 2, 3, 4]
squares = list(map(lambda x: x**2, numbers))

print(squares)
```

In this example, `map()` applies the lambda function `lambda x: x**2` to each element of the `numbers` list, producing a list of squares.

### 3.5 Using Lambda Functions with Pandas

Lambda functions are also commonly used in Pandas to apply operations across elements in a DataFrame. This is particularly useful for creating new columns or transforming data.

**Example:**

```{code-cell} ipython3
import pandas as pd

data = {"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]}
df = pd.DataFrame(data)

df["age_squared"] = df["age"].apply(lambda x: x**2)

print(df)
```

In this example, a Pandas DataFrame `df` is created with columns `name` and `age`. The `apply()` method is used to apply a lambda function that squares each element in the `age` column. The result is stored in a new column, `age_squared`.

### 3.6 Best Practices for Using Functions

1. **Use Descriptive Names:** Function names should clearly describe what the function does.
2. **Keep Functions Small and Focused:** A function should do one thing and do it well. If your function is getting too long, consider breaking it up into smaller functions.
3. **Document Your Functions:** Use docstrings to explain what your function does, what parameters it takes, and what it returns.

**Example of a Well-Documented Function:**

```python
def calculate_area(radius):
    """
    Calculate the area of a circle given its radius.

    Parameters:
    radius (float): The radius of the circle.

    Returns:
    float: The area of the circle.
    """
    return 3.14159 * radius ** 2
```

In this example, the function `calculate_area` is well-documented with a docstring that explains what the function does, its parameters, and its return value.

---

## Section 4: Hands-on Practice

Now that you've learned about control structures in Python, it's time to put your knowledge into practice. Below are a series of exercises designed to help reinforce the concepts you've covered. Each exercise includes a hint to guide you if you need a little help getting started.

### Exercise 1: Check if a Number is Even or Odd

```{admonition} Exercise 1
:class: exercise
Write a Python program to check whether a given number is even or odd.
```

```{dropdown} Hint
Use the modulo operator `%` to check if the number is divisible by `2`. If `number % 2 == 0`, the number is even; otherwise, it's odd.
```

### Exercise 2: Sum of All Numbers in a List

```{admonition} Exercise 2
:class: exercise
Write a Python program to find the sum of all the numbers in a list.
```

```{dropdown} Hint
Use a `for` loop to iterate over the elements in the list and accumulate the sum in a variable.
```

### Exercise 3: Factorial of a Number

```{admonition} Exercise 3
:class: exercise
Write a Python program to calculate the factorial of a given number.
```

```{dropdown} Hint
Use a `for` loop to iterate from `1` to `n`, multiplying the numbers together to calculate the factorial.
```

### Exercise 4: Check if a String is a Palindrome

```{admonition} Exercise 4
:class: exercise
Write a Python program to check if a given string is a palindrome.
```

```{dropdown} Hint
A palindrome reads the same forward and backward. Use string slicing to reverse the string and compare it to the original.
```

### Exercise 5: Find the Maximum and Minimum Elements in a List

```{admonition} Exercise 5
:class: exercise
Write a Python program to find the maximum and minimum elements in a list.
```

```{dropdown} Hint
Use the built-in `max()` and `min()` functions to find the maximum and minimum values in the list.
```

---

### Additional Exercises

If you're looking to challenge yourself further, here are a few more exercises to practice your Python skills:

1. **Exercise 6: Count the Number of Vowels in a String**  
   Write a Python program to count the number of vowels (`a, e, i, o, u`) in a given string.

   ```{dropdown} Hint
   Use a `for` loop to iterate over each character in the string and check if it's a vowel using the `in` operator.
   ```

2. **Exercise 7: Merge Two Lists and Remove Duplicates**  
   Write a Python program to merge two lists and remove any duplicate elements.

   ```{dropdown} Hint
   Combine the lists using the `+` operator, then convert the result to a `set` to remove duplicates before converting it back to a list.
   ```

3. **Exercise 8: Generate Fibonacci Sequence**  
   Write a Python program to generate the first `n` numbers of the Fibonacci sequence.

   ```{dropdown} Hint
   Use a `while` loop or a `for` loop to generate the sequence, starting with `0` and `1`, and adding the last two numbers to get the next number.
   ```

---

Feel free to explore these exercises further, and don’t hesitate to experiment with your own ideas. If you have any questions or need clarification, feel free to discuss them in the Slack channel. The more you practice, the more confident you’ll become in your Python programming skills. Good luck!
