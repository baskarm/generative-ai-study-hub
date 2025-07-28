
# 📊 Inferential Statistics & Hypothesis Testing

## 📌 Quick Navigation
- [1. Introduction to Inferential Statistics](#1-introduction-to-inferential-statistics)
- [2. Fundamental Terms in Distributions](#2-fundamental-terms-in-distributions)
- [3. Binomial Distribution](#3-binomial-distribution)
- [4. Uniform Distribution](#4-uniform-distribution)
- [5. Normal Distribution](#5-normal-distribution)
- [6. Z-Score](#6-z-score)
- [7. Sampling & Inference Foundations](#7-sampling--inference-foundations)
- [8. Central Limit Theorem](#8-central-limit-theorem)
- [9. Estimation](#9-estimation)
- [10. Hypothesis Testing](#10-hypothesis-testing)
- [CSV Download](#csv-download)
- [References & Further Reading](#references--further-reading)

---

## 1. Introduction to Inferential Statistics

Inferential statistics help us draw conclusions about populations based on sample data.

```python
# Example: confidence interval
import numpy as np
import scipy.stats as stats

sample = np.array([85, 80, 78, 90, 88])
conf_interval = stats.t.interval(0.95, len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))
print(conf_interval)
```

👉 [Open in Colab](https://drive.google.com/file/d/1I8KQJd-gg4-4aZ87ZpgYxpT4u-0q6gc0/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Use Case Example
📊 Used by analysts to infer population metrics from SAT sample scores.

[Back to Top](#-quick-navigation)

---

## 2. Fundamental Terms in Distributions

Covers mean, variance, standard deviation, skewness, and kurtosis.

### Use Case Example
📈 Financial institutions assess risk using variance and skewness of return distributions.

[Back to Top](#-quick-navigation)

---

## 3. Binomial Distribution

Applicable when analyzing binary outcomes (e.g., success/failure).

```python
from scipy.stats import binom

binom.pmf(k=3, n=10, p=0.5)
```

### Use Case Example
🧪 A/B testing for conversion rates on two landing pages.

[Back to Top](#-quick-navigation)

---

## 4. Uniform Distribution

All outcomes have equal probability.

### Use Case Example
🎲 Simulating random dice rolls or fair lottery draws.

[Back to Top](#-quick-navigation)

---

## 5. Normal Distribution

A bell-shaped distribution used across disciplines.

```python
from scipy.stats import norm

x = np.linspace(-3, 3, 100)
pdf = norm.pdf(x)
```

### Use Case Example
💡 Height distribution of people in a city, or standardized testing scores.

[Back to Top](#-quick-navigation)

---

## 6. Z-Score

Measures how many standard deviations a data point is from the mean.

```python
z = (x - np.mean(x)) / np.std(x)
```

### Use Case Example
🚨 Outlier detection in performance metrics.

[Back to Top](#-quick-navigation)

---

## 7. Sampling & Inference Foundations

Understanding population vs. sample, and designing sampling techniques.

### Use Case Example
🧬 Pharmaceutical companies conduct clinical trials on samples before full rollout.

[Back to Top](#-quick-navigation)

---

## 8. Central Limit Theorem

Describes how the sampling distribution of the sample mean approaches a normal distribution.

### Use Case Example
📉 Enables approximation of sampling behavior for metrics like average wait times.

[Back to Top](#-quick-navigation)

---

## 9. Estimation

Estimate population parameters like mean or proportion using sample statistics.

### Use Case Example
📊 Estimating average customer spend in a supermarket from sample receipt data.

[Back to Top](#-quick-navigation)

---

## 10. Hypothesis Testing

Formal process for testing claims using sample data.

👉 [Open in Colab](https://drive.google.com/file/d/1Wigcmj9mSDAsOCofSnT13A-0kgOFQNc3/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

```python
# Example: t-test
from scipy.stats import ttest_1samp

ttest_1samp(sample, popmean=85)
```

### Use Case Example
🧠 Determine if a new teaching method significantly improves test scores.

[Back to Top](#-quick-navigation)

---

## 📂 CSV Download

👉 [Download sat_score.csv](https://drive.google.com/uc?export=download&id=1Mc5hLwdsGD5qRup2nMJMi652ZgOSWyQV)  
📎 [View sat_score.csv](https://drive.google.com/file/d/1Mc5hLwdsGD5qRup2nMJMi652ZgOSWyQV/view)

👉 [Download debugging.csv](https://drive.google.com/uc?export=download&id=1H-1dVA9o-TLk0IdArDad3V2W7BnZPG22)  
📎 [View debugging.csv](https://drive.google.com/file/d/1H-1dVA9o-TLk0IdArDad3V2W7BnZPG22/view)

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

## 📄 PDF References

📘 [Inferential Statistics – Lecture PDF](../../pdfs/Lecture Slides -  Inferential Statistics.pdf)  
📘 [Hypothesis Testing – Lecture PDF](../../pdfs/Lecture Slides - Hypothesis Testing.pdf)

These PDFs are stored locally in your project under:
- `/docs/pdfs/Lecture Slides -  Inferential Statistics.pdf`
- `/docs/pdfs/Lecture Slides - Hypothesis Testing.pdf`

Ensure the paths align with your MkDocs file structure and navigation.

[Back to Top](#-quick-navigation)
---

**← Previous:** [Descriptive Statistics](01-descriptive-statistics.md)  
**→ Next:** [Distributions](03-distributions.md)