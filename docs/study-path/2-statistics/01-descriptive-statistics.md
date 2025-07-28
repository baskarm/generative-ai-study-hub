
# 📊 Descriptive Statistics – A Practical Guide

This lesson provides a detailed, example-rich walkthrough of key **Descriptive Statistics** concepts using Python. It includes real-world data analysis using Pandas, NumPy, and Seaborn, and is designed for students and data practitioners using the MkDocs Material theme.

---

## 📌 Quick Navigation

- [1. Introduction to Descriptive Statistics](#1-introduction-to-descriptive-statistics)
- [2. Dataset Overview](#2-dataset-overview)
- [3. Central Tendency Measures](#3-central-tendency-measures)
- [4. Variability Measures](#4-variability-measures)
- [5. Group-wise Descriptive Analysis](#5-group-wise-descriptive-analysis)
- [6. CSV Download](#6-csv-download)
- [7. Colab Notebook](#7-colab-notebook)
- [8. References & Further Reading](#8-references--further-reading)

---

## 1. Introduction to Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset in a quantitative manner. They form the foundation of exploratory data analysis (EDA).

---

## 2. Dataset Overview

This dataset (`descriptive_statistics_sample.csv`) contains sample records with `ID`, `Age`, `Income`, `SatisfactionScore`, and `PurchaseFrequency`.

---

## 3. Central Tendency Measures

### Measures Covered:
- **Mean**
- **Median**
- **Mode**

```python
import pandas as pd
data = pd.read_csv('descriptive_statistics_sample.csv')
print("Mean Age:", data['Age'].mean())
print("Median Age:", data['Age'].median())
print("Mode Age:", data['Age'].mode()[0])
```

👉 [Open in Colab](https://drive.google.com/file/d/10_XZByP73DdSx-uBkilJqO045FBK2IrQ/view?usp=sharing)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/10_XZByP73DdSx-uBkilJqO045FBK2IrQ/view?usp=sharing)

---

## 4. Variability Measures

### Measures Covered:
- **Range**
- **Variance**
- **Standard Deviation**

```python
range_income = data['Income'].max() - data['Income'].min()
print("Income Range:", range_income)
print("Income Variance:", data['Income'].var())
print("Income Std Dev:", data['Income'].std())
```

---

## 5. Group-wise Descriptive Analysis

Analyze descriptive stats by group (e.g., age group or satisfaction score).

```python
age_bins = pd.cut(data['Age'], bins=[18, 30, 45, 60], labels=['18-30', '31-45', '46-60'])
grouped = data.groupby(age_bins)['PurchaseFrequency'].mean()
print(grouped)
```

---

## 6. CSV Download

👉 [Download CSV from Google Drive](https://drive.google.com/uc?export=download&id=1yuiE5geF2jEjWmwV4Ye2jdKOqGYHsIP6)  
📎 [View CSV in Google Drive](https://drive.google.com/file/d/1yuiE5geF2jEjWmwV4Ye2jdKOqGYHsIP6/view)

[Back to Top](#-quick-navigation)

---

## 7. Colab Notebook

👉 [Open Notebook in Google Colab](https://drive.google.com/file/d/10_XZByP73DdSx-uBkilJqO045FBK2IrQ/view?usp=sharing)  
📎 [View on Google Drive](https://drive.google.com/file/d/10_XZByP73DdSx-uBkilJqO045FBK2IrQ/view)

---

## 8. References & Further Reading

- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Seaborn Docs](https://seaborn.pydata.org/)
- [Matplotlib Docs](https://matplotlib.org/stable/contents.html)
- [Khan Academy: Statistics](https://www.khanacademy.org/math/statistics-probability)
- [MIT OCW – Stats Courses](https://ocw.mit.edu/)
- [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer)
- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---
---

**→ Next:** [Inferential Statistics](02-inferential-statistics.md)