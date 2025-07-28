
# 🧠 Practical Exercise: Study 2 – Uber Demand Pattern Analysis

## 📌 Table of Contents
- [1. Overview](#1-overview)
- [2. Dataset Description](#2-dataset-description)
- [3. Methodology](#3-methodology)
- [4. Python Implementation](#4-python-implementation)
- [5. Insights & Interpretation](#5-insights--interpretation)
- [6. Use Case Impact](#6-use-case-impact)
- [7. CSV Download](#7-csv-download)
- [8. References & Further Reading](#8-references--further-reading)

---

## 1. Overview

### Context

Uber Technologies, Inc. is an American multinational transportation network company based in San Francisco and has operations in approximately 72 countries and 10,500 cities. In the fourth quarter of 2021, Uber had 118 million monthly active users worldwide and generated an average of 19 million trips per day.

Ridesharing is a very volatile market and demand fluctuates wildly with time, place, weather, local events, etc. The key to being successful in this business is to be able to detect patterns in these fluctuations and cater to demand at any given time.

As a newly hired Data Scientist in Uber's New York Office, you have been given the task of extracting insights from data that will help the business better understand the demand profile and take appropriate actions to drive better outcomes for the business. Your goal is to identify good insights that are potentially actionable, i.e., the business can do something with it.

### Objective

To extract actionable insights around demand patterns across various factors.

---

## 2. Dataset Description

The dataset contains information about the weather, location, and pickups. Below is the data dictionary:

- **pickup_dt**: Date and time of the pick-up  
- **borough**: NYC's borough  
- **pickups**: Number of pickups for the period  
- **spd**: Wind speed in miles/hour  
- **vsb**: Visibility in miles to the nearest tenth  
- **temp**: Temperature in Fahrenheit  
- **dewp**: Dew point in Fahrenheit  
- **slp**: Sea level pressure  
- **pcp01**: 1-hour liquid precipitation  
- **pcp06**: 6-hour liquid precipitation  
- **pcp24**: 24-hour liquid precipitation  
- **sd**: Snow depth in inches  
- **hday**: Being a holiday (Y) or not (N)

---

## 3. Methodology

### Key Questions

1. What are the different variables that influence pickups?  
2. Which factor affects the pickups the most? What could be plausible reasons for that?  
3. What are your recommendations to Uber management to capitalize on fluctuating demand?

### Guidelines

- Perform **univariate analysis** to better understand individual variables.
- Perform **bivariate analysis** to explore relationships between variables.
- Create **visualizations** to explore the data and extract actionable insights.

---

## 4. Python Implementation

👉 [Open in Colab](https://drive.google.com/file/d/1d0kAd4fmDl7rW8IY5qnX8KS2H-vtbUrP/view?usp=sharing)  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1d0kAd4fmDl7rW8IY5qnX8KS2H-vtbUrP/view?usp=sharing)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Uber.csv")
print(df.head())

# Example: Analyze pickups by borough
sns.boxplot(data=df, x='borough', y='pickups')
plt.title("Distribution of Pickups by NYC Borough")
plt.xticks(rotation=45)
plt.show()
```

---

## 5. Insights & Interpretation

- **Borough-wise Patterns**: Some boroughs consistently show higher ride demand.
- **Weather Influence**: Variables like temperature and precipitation correlate with demand.
- **Holiday Trends**: Holidays (hday = Y) may show increased or decreased pickup patterns depending on context.

Visualizations and full interpretation are included in the Google Colab notebook.

---

## 6. Use Case Impact

📊 These insights help Uber to:
- Adjust pricing or fleet placement by borough
- Forecast ride surges during bad weather or holidays
- Optimize driver incentives during low-visibility conditions

---

## 7. CSV Download

👉 [Download Uber.csv](https://drive.google.com/uc?export=download&id=1hcAZNHw3HQ50o0JFcgpQ5JAMTzDHhXe4)

📎 [View Uber.csv in Google Drive](https://drive.google.com/file/d/1hcAZNHw3HQ50o0JFcgpQ5JAMTzDHhXe4/view)

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

[Back to Top](#-table-of-contents)
---
[⬅️ Previous: CardioGood Fitness Case Study](08-practical-exercise-study-1-cardiogood-fitness.md) | [Next: Exploratory Data Analysis ➡️](10-exploratorydataanalysis-EDA.md)
---