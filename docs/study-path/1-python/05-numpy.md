
# 🧮 Chapter 1.1: NumPy

## 📌 Quick Navigation
- [1. Introduction to NumPy](#1-introduction-to-numpy)
- [2. NumPy Functions](#2-numpy-functions)
- [3. Accessing NumPy Array](#3-accessing-numpy-array)
- [4. Modifying the Entries of a Matrix](#4-modifying-the-entries-of-a-matrix)
- [5. Saving and Loading NumPy Arrays](#5-saving-and-loading-numpy-arrays)
- [References & Further Reading](#references--further-reading)

---

## 1. Introduction to NumPy

NumPy (Numerical Python) is a core Python library for scientific computing, offering fast, flexible multi-dimensional `ndarray` objects and a suite of mathematical tools([numpy.org](https://numpy.org/devdocs//user/absolute_beginners.html?utm_source=chatgpt.com), [w3schools.com](https://www.w3schools.com/python/numpy/default.asp?utm_source=chatgpt.com), [stackoverflow.com](https://stackoverflow.com/questions/60199316/how-to-save-a-list-of-numpy-arrays-into-a-single-file-and-load-file-back-to-orig?utm_source=chatgpt.com), [geeksforgeeks.org](https://www.geeksforgeeks.org/python/introduction-to-numpy/?utm_source=chatgpt.com)).  
Arrays are stored in contiguous memory, enabling vectorized operations that run orders of magnitude faster than pure Python loops([geeksforgeeks.org](https://www.geeksforgeeks.org/python/introduction-to-numpy/?utm_source=chatgpt.com)).

```python
import numpy as np

array = np.array([1, 2, 3])
print(array)
```

👉 [Open in Colab](https://colab.research.google.com/drive/1GaQw5p0XCLVk299L4QSJiUJOE0vVyX0y?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
📊 NumPy is used in financial analysis for high-speed matrix operations when simulating stock market behaviors.

[Back to Top](#-quick-navigation)

---

## 2. NumPy Functions

Common NumPy functions include reshaping, random number generation, mathematical operations, and broadcasting.

```python
a = np.arange(10).reshape(2, 5)
b = np.random.rand(2, 5)
print(np.add(a, b))
```

### Use Case Example
🧠 Neuroscience labs use NumPy to simulate electrical signals across neurons represented in matrix form.

[Back to Top](#-quick-navigation)

---

## 3. Accessing NumPy Array

Accessing elements is done with indexing and slicing.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])  # Output: 2
```

### Use Case Example
📈 Sports analysts extract specific performance metrics from multidimensional datasets using array indexing.

[Back to Top](#-quick-navigation)

---

## 4. Modifying the Entries of a Matrix

You can change values in NumPy arrays directly using indexing or conditions.

```python
matrix = np.array([[1, 2], [3, 4]])
matrix[0, 1] = 10
print(matrix)
```

### Use Case Example
🧬 Bioinformaticians adjust gene expression data arrays during cleaning and normalization stages.

[Back to Top](#-quick-navigation)

---

## 5. Saving and Loading NumPy Arrays

Use `.npy` and `.npz` formats for saving and restoring arrays efficiently.

```python
np.save("my_array.npy", arr)
loaded = np.load("my_array.npy")
```

### Use Case Example
📂 Machine learning engineers store preprocessed input arrays for quick reuse across experiments.

[Back to Top](#-quick-navigation)

---

## 🔗 References & Further Reading

- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Seaborn Docs](https://seaborn.pydata.org/)
- [Plotly Python Docs](https://plotly.com/python/)
- [Khan Academy: Statistics & Probability](https://www.khanacademy.org/math/statistics-probability)
- [MIT OpenCourseWare – Stats](https://ocw.mit.edu/)
- [StatQuest by Josh Starmer (YouTube)](https://www.youtube.com/user/joshstarmer)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

[Back to Top](#-quick-navigation)
---
[⬅️ Previous: Functions](04-functions.md) | [Next: Pandas ➡️](06-pandas.md)
---