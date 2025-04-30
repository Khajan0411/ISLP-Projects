
# Chapter 2: Introduction to Python (ISLP Lab)
# ---------------------------------------------

# Basic Python commands
x = 3
print("x =", x)
y = 4
print("y =", y)
print("x + y =", x + y)

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from islp import load_data

# NumPy basics
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("a + b =", a + b)

# Indexing and slicing
arr = np.arange(10)
print("arr[3:7] =", arr[3:7])

# Matplotlib example
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()

# Load and inspect data using ISLP
Auto = load_data('Auto')
print("Auto dataset head:")
print(Auto.head())

# Summary statistics
print("Summary statistics of 'mpg':")
print(Auto['mpg'].describe())

# Plotting data
plt.scatter(Auto['horsepower'], Auto['mpg'])
plt.title("Horsepower vs. MPG")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()
