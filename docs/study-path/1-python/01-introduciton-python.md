
# 🧠 Statistics & Data Science: Python Introduction

## 📌 Quick Navigation
- [1. Print Statements](#1-print-statements)
- [2. Variables](#2-variables)
- [3. Data Types](#3-data-types)
- [4. Basic Operators](#4-basic-operators)
- [5. Data Structures – List](#5-data-structures--list)
- [6. Data Structures – Tuple](#6-data-structures--tuple)
- [7. Data Structures – Dictionary](#7-data-structures--dictionary)
- [8. Type Checking](#8-type-checking)
- [9. Comparison Operators](#9-comparison-operators)
- [10. Conditional Statements](#10-conditional-statements)
- [12. References & Further Reading](#references--further-reading)

---

## 1. Print Statements

- Python’s `print()` function is used to display output.

```python
print('The name of the company is Cars Sons Ltd.')
print('The year the company was established is', 1996)
print('Total turnover of the company this year in Million $ is', 12.5)
```

### Use Case Example
Essential for logging, debugging, and showing results of analysis.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 2. Variables

- Containers for storing data values.

```python
company_name = "Cars Sons Ltd."
year_started = 1996
turnover = 12.5
```

### Use Case Example
Used to store company attributes, which feed into models or business dashboards.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 3. Data Types

- `int`, `float`, `str` are common types.

```python
type(company_name)  # str
type(year_started)  # int
type(turnover)      # float
```

### Use Case Example
Understanding data types is essential for error-free data preprocessing.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 4. Basic Operators

- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**`

```python
price_sedan = 0.2
total_price = price_sedan * 10
```

### Use Case Example
Used in calculating sales metrics or KPIs in business analytics.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 5. Data Structures – List

- Lists store multiple items.

```python
cars = ['Sedan', 'SUV', 'Hatchback']
sales = [200, 400, 300]
```

### Use Case Example
Track car types or regional sales in customer segmentation.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 6. Data Structures – Tuple

- Immutable ordered collections.

```python
sitting = (5, 4, 6)
```

### Use Case Example
Store constant metadata like seat capacity or configuration specs.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 7. Data Structures – Dictionary

- Key-value pairs for structured data.

```python
sales_last_month = {'Sedan': 2, 'SUV': 1.5}
```

### Use Case Example
Used for storing car-wise sales figures, performance ratings, etc.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 8. Type Checking

```python
type(sales_last_month)  # dict
```

### Use Case Example
Validating type before transformations in ETL workflows.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 9. Comparison Operators

- Compare data values: `==`, `!=`, `<`, `>`, `<=`, `>=`

```python
sales['Sedan'] > sales['Hatchback']
```

### Use Case Example
Used in dashboards to compare performance trends.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
[Back to Top](#-quick-navigation)

---

## 10. Conditional Statements

- Use `if`, `else` to apply logic

```python
if cars_ratings['Sedan_1'] > 50:
    print("Good performer")
else:
    print("Needs improvement")
```

### Use Case Example
Flag underperforming models for review.

👉 [Open in Colab](https://colab.research.google.com/drive/1JngriMWczFKWlD-sXI5Q7lWhbt0Aa2N_?usp=sharing)  
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


➡️ [Next: Collection of Variables](02-collection-of-variables.md)