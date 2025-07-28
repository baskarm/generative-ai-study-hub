
# 📊 Chapter 1.3: Data Visualization

## 📌 Quick Navigation
- [1. Data Loading and Overview](#1-data-loading-and-overview)
- [2. Histogram](#2-histogram)
- [3. Histogram with Density Curve](#3-histogram-with-density-curve)
- [4. Box Plot](#4-box-plot)
- [5. Line Plot](#5-line-plot)
- [6. Scatter Plot](#6-scatter-plot)
- [7. lm Plot in Seaborn](#7-lm-plot-in-seaborn)
- [8. Swarm Plot](#8-swarm-plot)
- [9. Pair Plot](#9-pair-plot)
- [10. Heat Map](#10-heat-map)
- [11. Plotly](#11-plotly)
- [12. Customizing Plots](#12-customizing-plots)
- [CSV Download](#csv-download)
- [References & Further Reading](#references--further-reading)

---

## 1. Data Loading and Overview

Data loading is the first step in any data visualization or analysis pipeline. It involves reading data from CSV, Excel, or APIs using libraries like `pandas`.

- Use `pandas.read_csv()` to load CSV files.
- Understand data types, null values, and summary statistics.

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
df.info()
df.describe()
```

👉 [Open in Colab](https://colab.research.google.com/drive/1pF4W7kAzzS63jGhbWA8C4OKGgiuljLJs?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
📈 In healthcare, loading large hospital datasets with patient records allows data scientists to quickly inspect missing values and perform visual triage of data quality.

[Back to Top](#-quick-navigation)

---

## 2. Histogram

Histograms display the distribution of numerical data by grouping values into bins.

- Great for identifying skewness, spread, and outliers.
- Created using `seaborn.histplot()` or `matplotlib.pyplot.hist()`.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["age"], bins=10)
plt.show()
```

👉 [Open in Colab](https://colab.research.google.com/drive/1pF4W7kAzzS63jGhbWA8C4OKGgiuljLJs?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
🎓 Universities use histograms to visualize student grade distributions across departments.

[Back to Top](#-quick-navigation)

---

## 3. Histogram with Density Curve

Combining histograms with KDE (Kernel Density Estimation) overlays helps visualize probability density.

```python
sns.histplot(df["income"], kde=True)
```

### Use Case Example
💰 Financial analysts use density curves over income brackets to detect abnormal income distributions for fraud detection.

[Back to Top](#-quick-navigation)

---

## 4. Box Plot

Box plots summarize the distribution using median, quartiles, and outliers.

```python
sns.boxplot(x="region", y="salary", data=df)
```

### Use Case Example
🏥 Hospitals compare patient waiting times by department using box plots to identify service bottlenecks.

[Back to Top](#-quick-navigation)

---

## 5. Line Plot

Line plots show trends over time or ordered categories.

```python
sns.lineplot(x="year", y="revenue", data=df)
```

### Use Case Example
📉 Economists visualize GDP trends over decades with line plots.

[Back to Top](#-quick-navigation)

---

## 6. Scatter Plot

Scatter plots reveal relationships between two numeric variables.

```python
sns.scatterplot(x="height", y="weight", data=df)
```

### Use Case Example
📊 Insurance companies use scatter plots to assess correlations between age and insurance claims.

[Back to Top](#-quick-navigation)

---

## 7. lm Plot in Seaborn

lm plots add regression lines to scatter plots, useful for trend analysis.

```python
sns.lmplot(x="experience", y="salary", data=df)
```

### Use Case Example
💼 HR departments forecast salary expectations based on years of experience.

[Back to Top](#-quick-navigation)

---

## 8. Swarm Plot

Swarm plots show all data points and avoid overlapping unlike box plots.

```python
sns.swarmplot(x="department", y="satisfaction", data=df)
```

### Use Case Example
🧪 Clinical research teams visualize patient responses to different treatments using swarm plots.

[Back to Top](#-quick-navigation)

---

## 9. Pair Plot

Pair plots visualize relationships across multiple variables in one grid.

```python
sns.pairplot(df[["age", "income", "expenses"]])
```

### Use Case Example
📊 Marketing teams explore consumer segmentation by analyzing patterns in spending behavior.

[Back to Top](#-quick-navigation)

---

## 10. Heat Map

Heatmaps visualize matrix-like data using color intensities.

```python
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
```

### Use Case Example
🔍 Data scientists use heatmaps to visualize correlation matrices for feature selection in ML pipelines.

[Back to Top](#-quick-navigation)

---

## 11. Plotly

Plotly provides interactive, zoomable, and publishable charts in Python.

```python
import plotly.express as px
px.scatter(df, x="age", y="income", color="region")
```

### Use Case Example
📊 News outlets use Plotly to create interactive COVID-19 dashboards for public engagement.

[Back to Top](#-quick-navigation)

---

## 12. Customizing Plots

Customize color palettes, themes, fonts, and axis labels for clearer visuals.

```python
sns.set(style="whitegrid", palette="pastel")
sns.boxplot(x="gender", y="score", data=df)
```

### Use Case Example
📢 In presentations, using color-blind-friendly palettes ensures accessibility for all stakeholders.

[Back to Top](#-quick-navigation)

---

## 📂 CSV Download

👉 [Download CSV from Google Drive](https://drive.google.com/uc?export=download&id=1tWVw9Zd8PzX1GE29KOPxRJCKS9b3w37v)

📎 [View CSV in Google Drive](https://drive.google.com/file/d/1tWVw9Zd8PzX1GE29KOPxRJCKS9b3w37v/view)

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
[⬅️ Previous: Pandas](06-pandas.md) | [Next: CardioGood Fitness Case Study ➡️](08-practical-exercise-study-1-cardiogood-fitness.md)
---