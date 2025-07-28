
# 🧠 Statistics & Data Science: Control Flow in Python

## 📌 Quick Navigation
- [1. Conditional Statements](#1-conditional-statements)
- [2. Looping in Python](#2-looping-in-python)
  - [2.1 For Loop](#21-for-loop)
  - [2.2 While Loop](#22-while-loop)
  - [2.3 For Loop with Conditional Statements](#23-for-loop-with-conditional-statements)
  - [2.4 For Loop with Accumulation](#24-for-loop-with-accumulation)
  - [2.5 For Loop with Filtering](#25-for-loop-with-filtering)
  - [2.6 For Loop with Character Iteration](#26-for-loop-with-character-iteration)
  - [2.7 For Loop with Repetition](#27-for-loop-with-repetition)
- [References & Further Reading](#references--further-reading)

---

## 1. Conditional Statements

- Conditional statements allow programs to make decisions based on conditions.
- Commonly used keywords: `if`, `elif`, `else`.

```python
# Example: Check if a number is even or odd
a = 2020
if a % 2 == 0:
    print("Even")
else:
    print("Odd")
```

👉 [Open in Colab](https://colab.research.google.com/drive/1BGdRSAQTKdgHmnEMGKrHLY73s6d9kWh0?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
Conditional statements are essential for business logic, e.g., evaluating credit scores to approve/reject a loan.

[Back to Top](#-quick-navigation)

---

## 2. Looping in Python

Loops allow you to repeat a block of code multiple times.

---

### 2.1 For Loop

- Used to iterate over a sequence (list, tuple, range, etc.)

```python
for i in range(5):
    print(i)
```

---

### 2.2 While Loop

- Repeats as long as a condition is true

```python
i = 1
while i <= 5:
    print(i)
    i += 1
```

---

### 2.3 For Loop with Conditional Statements

```python
for i in range(1, 11):
    if i % 2 == 0:
        print(f"{i} is even")
```

---

### 2.4 For Loop with Accumulation

```python
total = 0
for i in range(1, 101):
    total += i
print(total)
```

---

### 2.5 For Loop with Filtering

```python
for i in range(1, 51):
    if i % 7 == 0 and i % 5 != 0:
        print(i)
```

---

### 2.6 For Loop with Character Iteration

```python
st = "Data Science"
for char in st:
    print(char)
```

---

### 2.7 For Loop with Repetition

```python
for _ in range(5):
    print("Data Science")
```

👉 [Open in Colab](https://colab.research.google.com/drive/1DIDoeKMbDjGtfyh61A3t9Wqw--Pulivi?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
Looping constructs are widely used in automation, report generation, and simulations in data analysis workflows.

[Back to Top](#-quick-navigation)

---

## References & Further Reading

- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Seaborn Docs](https://seaborn.pydata.org/)
- [Plotly Python](https://plotly.com/python/)
- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- [MIT OpenCourseWare](https://ocw.mit.edu/)
- [StatQuest (YouTube)](https://www.youtube.com/user/joshstarmer)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
---
[⬅️ Previous: Collection of Variables](02-collection-of-variables.md) | [Next: Functions ➡️](04-functions.md)
---