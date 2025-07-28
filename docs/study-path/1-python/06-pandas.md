
# 🐼 Chapter 1.2: Pandas

## 📌 Quick Navigation
- [1. Introduction to Pandas](#1-introduction-to-pandas)
- [2. Accessing Series and DataFrames](#2-accessing-series-and-dataframes)
- [3. loc and iloc in Pandas](#3-loc-and-iloc-in-pandas)
- [4. Condition-Based Indexing](#4-condition-based-indexing)
- [5. Combining DataFrames](#5-combining-dataframes)
- [6. Saving and Loading DataFrames](#6-saving-and-loading-dataframes)
- [7. Statistical Functions](#7-statistical-functions)
- [8. GroupBy Function](#8-groupby-function)
- [9. Date and Time Functions](#9-date-and-time-functions)
- [References & Further Reading](#references--further-reading)

---

## 1. Introduction to Pandas

Pandas is a powerful Python library for data manipulation and analysis. It provides data structures like Series and DataFrames.

```python
import pandas as pd

data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)
```

👉 [Open in Colab](https://colab.research.google.com/drive/1O9FIzLybtd6OOYuk75wBQS_H9Yk4iAzR?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
🏥 Hospitals use Pandas to store and manipulate patient records for analysis and visualization.

[Back to Top](#-quick-navigation)

---

## 2. Accessing Series and DataFrames

Pandas Series is a one-dimensional labeled array, and DataFrame is a two-dimensional table.

```python
series = df["Age"]
print(series)
print(type(series))
```

### Use Case Example
💼 HR analysts retrieve employee age or salary columns to perform segmentation.

[Back to Top](#-quick-navigation)

---

## 3. loc and iloc in Pandas

- `.loc[]` accesses rows by label.
- `.iloc[]` accesses rows by index position.

```python
print(df.loc[0])   # Access by label
print(df.iloc[1])  # Access by index
```

### Use Case Example
📊 Researchers extract specific survey responses using row labels or positions.

[Back to Top](#-quick-navigation)

---

## 4. Condition-Based Indexing

Filter DataFrames using Boolean conditions.

```python
df[df["Age"] > 26]
```

### Use Case Example
🏢 Companies filter customers by age for targeted advertising campaigns.

[Back to Top](#-quick-navigation)

---

## 5. Combining DataFrames

You can concatenate or merge multiple DataFrames.

```python
df1 = pd.DataFrame({"ID": [1, 2], "Name": ["Alice", "Bob"]})
df2 = pd.DataFrame({"ID": [1, 2], "Age": [25, 30]})
merged = pd.merge(df1, df2, on="ID")
```

### Use Case Example
📦 Merging order and customer data allows fulfillment centers to improve logistics tracking.

[Back to Top](#-quick-navigation)

---

## 6. Saving and Loading DataFrames

Save and load datasets using CSV or Excel formats.

```python
df.to_csv("output.csv", index=False)
df_loaded = pd.read_csv("output.csv")
```

### Use Case Example
💽 Analysts export cleaned datasets to share with business teams or dashboards.

[Back to Top](#-quick-navigation)

---

## 7. Statistical Functions

Pandas includes descriptive statistics like mean, median, std, etc.

```python
df["Age"].mean()
df.describe()
```

### Use Case Example
📉 Healthcare analysts summarize patient vitals and lab results for reports.

[Back to Top](#-quick-navigation)

---

## 8. GroupBy Function

Group data by a categorical variable and apply aggregate functions.

```python
df.groupby("Name")["Age"].mean()
```

### Use Case Example
🛍️ Retailers group transactions by store to compute average revenue per store.

[Back to Top](#-quick-navigation)

---

## 9. Date and Time Functions

Convert and manipulate datetime formats in Pandas.

```python
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
```

### Use Case Example
📅 Analysts break down sales by month or quarter using datetime fields.

[Back to Top](#-quick-navigation)

---


---

## 📂 CSV Download

👉 [Download StockData.csv](https://drive.google.com/uc?export=download&id=1tWn3Rd2t2BnPYi5bwAmi-joRjh06t_AH)

📎 [View StockData.csv in Google Drive](https://drive.google.com/file/d/1tWn3Rd2t2BnPYi5bwAmi-joRjh06t_AH/view)

[Back to Top](#-quick-navigation)


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
[⬅️ Previous: NumPy](05-numpy.md) | [Next: Data Visualization ➡️](07-data-visualization.md)
---