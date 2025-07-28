# 🧠 Statistics & Data Science – Python Functions

## 📌 Quick Navigation
- [1. Basic Arithmetic Function](#1-basic-arithmetic-function)
- [2. Squares in Range](#2-squares-in-range)
- [3. Simple Interest Calculation](#3-simple-interest-calculation)
- [4. Divisibility by 25](#4-divisibility-by-25)
- [5. Square and Add Five](#5-square-and-add-five)
- [6. Lambda: Square and Add Five](#6-lambda-square-and-add-five)
- [7. Power Function](#7-power-function)
- [8. Triangle Area](#8-triangle-area)
- [9. Country Origin Function](#9-country-origin-function)
- [10. Celsius to Fahrenheit](#10-celsius-to-fahrenheit)

---

## 1. Basic Arithmetic Function

Define a function to add, subtract, multiply, and divide two variables.

```python
def perform_operations(a, b):
    print(f"Addition: {a + b}")
    print(f"Subtraction: {a - b}")
    print(f"Multiply: {a * b}")
    if b != 0:
        print(f"Division: {a / b}")
    else:
        print("Division by zero is not allowed.")

perform_operations(10, 5)
```

👉 [Open in Colab](https://colab.research.google.com/drive/1QyRsIceRsdYpkNcFw17HTthAe1w00BXo?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

[Back to Top](#-quick-navigation)

## 2. Squares in Range

Function to print squares of numbers from 1 to 10.

```python
def print_squares_in_range():
    for i in range(1, 11):
        print(i * i)

print_squares_in_range()
```

[Back to Top](#-quick-navigation)

## 3. Simple Interest Calculation

Formula:  
$SI = \frac{P \times R \times T}{100}$

```python
def calculate_simple_interest(principal, rate, time):
    return (principal * rate * time) / 100

calculate_simple_interest(1000, 3, 5)
```

[Back to Top](#-quick-navigation)

## 4. Divisibility by 25

```python
def is_divisible_by_25(number):
    return True if number % 25 == 0 else "Not divisible"
```

[Back to Top](#-quick-navigation)

## 5. Square and Add Five

```python
def square_and_add_five(number):
    return number ** 2 + 5
```

[Back to Top](#-quick-navigation)

## 6. Lambda: Square and Add Five

```python
square_add_five_lambda = lambda x: x ** 2 + 5
```

[Back to Top](#-quick-navigation)

## 7. Power Function

```python
def calculate_power(base, exponent):
    return base ** exponent
```

[Back to Top](#-quick-navigation)

## 8. Triangle Area

Formula:  
$Area = \frac{1}{2} \times base \times height$

```python
def calculate_triangle_area(base, height):
    return 0.5 * base * height
```

[Back to Top](#-quick-navigation)

## 9. Country Origin Function

```python
def print_country_origin(country):
    return f"I am from {country}"
```

[Back to Top](#-quick-navigation)

## 10. Celsius to Fahrenheit

Formula:  
$F = \frac{9}{5}C + 32$

```python
def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32
```

[Back to Top](#-quick-navigation)

---

## References & Further Reading

- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Seaborn Docs](https://seaborn.pydata.org/)
- [Plotly Docs](https://plotly.com/python/)
- [Khan Academy - Statistics](https://www.khanacademy.org/math/statistics-probability)
- [MIT OCW](https://ocw.mit.edu/)
- [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
---
[⬅️ Previous: Conditional & Looping](03-conditional-looping.md) | [Next: NumPy ➡️](05-numpy.md)
---