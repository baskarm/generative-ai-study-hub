
# 🧠 Case Study: CardioGood Fitness Data Analysis

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

### Problem Statement – CardioGood Fitness Data Analysis

**Context:**  
The market research team at **AdRight** is assigned the task to identify the profile of the typical customer for each treadmill product offered by **CardioGood Fitness**. The market research team decides to investigate whether there are differences across the product lines with respect to customer characteristics. The team decides to collect data on individuals who purchased a treadmill at a **CardioGood Fitness** retail store at any time in the past three months. The data is stored in the `CardioGoodFitness.csv` file.

**Objective:**  
Perform descriptive analysis to create a **customer profile** for each CardioGood Fitness treadmill product line.

---

## 2. Dataset Description

**Data Dictionary:**  
The team identified the following customer variables to study:

- **Product**: Product purchased - TM195, TM498, or TM798  
- **Gender**: Male or Female  
- **Age**: Age of the customer in years  
- **Education**: Education of the customer in years  
- **MaritalStatus**: Single or Partnered  
- **Income**: Annual household income  
- **Usage**: The average number of times the customer plans to use the treadmill each week  
- **Miles**: The average number of miles the customer expects to walk/run each week  
- **Fitness**: Self-rated fitness on a 1-to-5 scale, where 1 is poor shape and 5 is excellent shape

---

## 3. Methodology

### Questions to Explore:

1. What are the different types of variables in the data?  
2. What is the distribution of different variables in the data?  
3. Which product is more popular among males or females?  
4. Is the product purchase affected by the marital status of the customer?  
5. Is there a significant correlation among some of the variables?  
6. What is the distribution of the average number of miles for each product?

---

## 4. Python Implementation

👉 [Open in Colab](https://colab.research.google.com/drive/14rCNMJQRPqNaPflc2d7AuEGugLLsIHPq?usp=sharing)  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1RmV5pvUHZKKVcDSuPGS8udMqo75XZNFL/view?usp=sharing)

```python
# Sample code block from the solution notebook
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("CardioGoodFitness.csv")
print(df.head())

# Visualizing product preference by gender
sns.countplot(data=df, x='Product', hue='Gender')
plt.title("Product Preference by Gender")
plt.show()
```

---

## 5. Insights & Interpretation

Insights will be derived through descriptive analysis using data visualizations and statistical exploration in the Colab notebook. Key focus areas include:

- Product popularity across demographic segments
- Relationship between fitness and treadmill usage
- Impact of education, age, and income on product choice

Refer to the Google Colab notebook for detailed implementation and visuals.

---

## 6. Use Case Impact

🎯 This analysis helps the business:
- Define distinct customer personas per product line
- Design targeted marketing strategies
- Identify potential market gaps across demographics

---

## 7. CSV Download

👉 [Download CardioGoodFitness.csv](https://drive.google.com/uc?export=download&id=1wBEmNyahUN693Nv0Tdy_6e8eKxjj92Nt)

📎 [View CardioGoodFitness.csv](https://drive.google.com/file/d/1wBEmNyahUN693Nv0Tdy_6e8eKxjj92Nt/view)

📊 [View Solutions Data (XLSX)](https://docs.google.com/spreadsheets/d/146e-XgE7PIACFBUdZLa-4G-4pTb1ytAx/edit?usp=sharing)

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
[⬅️ Previous: Data Visualization](07-data-visualization.md) | [Next: Uber Case Study ➡️](09-practical-exercise-study-2-uber.md)
---