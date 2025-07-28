
# 📊 Exploratory Data Analysis (EDA)

## 📌 Quick Navigation
- [1. Data Loading and Initial Exploration](#1-data-loading-and-initial-exploration)
- [2. Checking Missing, Duplicate Values, and Summary](#2-checking-missing-duplicate-values-and-summary)
- [3. Univariate Analysis](#3-univariate-analysis)
- [4. Bivariate Analysis](#4-bivariate-analysis)
- [5. Charts and Plots for EDA](#5-charts-and-plots-for-eda)
- [6. Missing Values - Group Mean](#6-missing-values---group-mean)
- [7. Missing Values - Medians & Dropping](#7-missing-values---medians--dropping)
- [8. Outlier Detection and Analysis](#8-outlier-detection-and-analysis)
- [CSV Download](#csv-download)
- [References & Further Reading](#references--further-reading)

---

## 1. Data Loading and Initial Exploration

Begin by loading the housing dataset and performing initial inspection using Pandas.

```python
import pandas as pd

df = pd.read_csv("Melbourne_Housing.csv")
df.head()
```

👉 [Open in Colab](https://colab.research.google.com/drive/1dNmQclwIU50dT-dyth_tbIm6U-9Gymbs?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
🏘️ Real estate analysts begin their data pipeline with loading raw housing sales data to inspect types and formats.

[Back to Top](#quick-navigation)

---

## 2. Checking Missing, Duplicate Values, and Summary

Use `.isnull().sum()` to inspect missing values, and `.describe()` for a statistical summary.

```python
print(df.isnull().sum())
df.describe()
```

### Use Case Example
🔍 Detecting columns with substantial missing data helps decide cleaning strategy in housing market analytics.

[Back to Top](#quick-navigation)

---

## 3. Univariate Analysis

Study single variable distributions using histograms and value counts.

```python
df["Price"].hist(bins=50)
df["Type"].value_counts()
```

### Use Case Example
💲 Understand housing price distribution to identify outliers or skewness in Melbourne.

[Back to Top](#quick-navigation)

---

## 4. Bivariate Analysis

Use scatter plots and correlation matrices to explore relationships between variables.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x="Distance", y="Price", data=df)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
```

### Use Case Example
📉 Discover how distance from the city center influences real estate pricing.

[Back to Top](#quick-navigation)

---

## 5. Charts and Plots for EDA

Generate box plots, bar charts, pair plots, and violin plots for deeper insight.

```python
sns.boxplot(x="Type", y="Price", data=df)
```

### Use Case Example
📊 Visual analytics are used in real estate investment dashboards to summarize sales trends by property type.

[Back to Top](#quick-navigation)

---

## 6. Missing Values - Group Mean

Fill missing values with group-specific means.

```python
df["BuildingArea"].fillna(df.groupby("Type")["BuildingArea"].transform("mean"), inplace=True)
```

### Use Case Example
🏗️ Impute area details for residential properties based on type-specific averages.

[Back to Top](#quick-navigation)

---

## 7. Missing Values - Medians & Dropping

Alternative strategy for missing value treatment by median filling or row dropping.

```python
df["Bedroom2"].fillna(df["Bedroom2"].median(), inplace=True)
df.dropna(inplace=True)
```

### Use Case Example
📉 Reducing noise caused by sparse or unfixable data improves ML model training.

[Back to Top](#quick-navigation)

---

## 8. Outlier Detection and Analysis

Detect and remove outliers using IQR or z-score techniques.

```python
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1

filtered_df = df[(df["Price"] >= Q1 - 1.5 * IQR) & (df["Price"] <= Q3 + 1.5 * IQR)]
```

### Use Case Example
💡 Outlier removal leads to more stable forecasting of property valuation models.

[Back to Top](#quick-navigation)

---

## 📂 CSV Download

👉 [Download Melbourne_Housing.csv](https://drive.google.com/uc?export=download&id=1AoxHXwfjLHpIqsrScMrChm2BGrz5D-QJ)  
📎 [View Melbourne_Housing.csv](https://drive.google.com/file/d/1AoxHXwfjLHpIqsrScMrChm2BGrz5D-QJ/view)

👉 [Download Melbourne_Housing_NoMissing.csv](https://drive.google.com/uc?export=download&id=17EFiplaWGltOGhXFBi2LVwSGKvqr6mBw)  
📎 [View Melbourne_Housing_NoMissing.csv](https://drive.google.com/file/d/17EFiplaWGltOGhXFBi2LVwSGKvqr6mBw/view)

👉 [Download Melbourne_Housing_NoOutliers.csv](https://drive.google.com/uc?export=download&id=1HYpX2N7kIY_rNJ25EX2hB4s8VPnRfzcq)  
📎 [View Melbourne_Housing_NoOutliers.csv](https://drive.google.com/file/d/1HYpX2N7kIY_rNJ25EX2hB4s8VPnRfzcq/view)

[Back to Top](#quick-navigation)

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

[Back to Top](#quick-navigation)
---
[⬅️ Previous: Uber Case Study](09-practical-exercise-study-2-uber.md)
---