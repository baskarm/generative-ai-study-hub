# 📊 Probability Distributions & Inferential Statistics

## 📌 Quick Navigation
- [1. Binomial Distribution](#1-binomial-distribution)
- [2. Uniform Distribution](#2-uniform-distribution)
- [3. Normal Distribution](#3-normal-distribution)
- [4. Sampling Distribution](#4-sampling-distribution)
- [5. Interval Estimation](#5-interval-estimation)
- [6. Hypothesis Testing](#6-hypothesis-testing)
- [7. CSV Download](#7-csv-download)
- [8. References & Further Reading](#8-references--further-reading)

---

## 1. Binomial Distribution

- Describes number of successes in a fixed number of independent Bernoulli trials.
- Each trial has only two outcomes: success or failure.
- Example formula: \( P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} \)

### Use Case Example
- **Marketing Campaign**: Predicting number of people who will respond positively to an ad campaign out of 1000 targeted.
- **Real-time News**: Email A/B testing for response rate based on different subject lines.

👉 [Open in Colab](https://drive.google.com/file/d/1-MYpXtK3qodGp9g-HHbXI73lmiiey7bw/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

[Back to Top](#-quick-navigation)

---

## 2. Uniform Distribution

- All outcomes are equally likely within the range \( [a, b] \).
- Used in simulations or modeling flat/no-preference systems.

### Use Case Example
- **Simulation Models**: Randomized control group assignment.
- **Real-time News**: Modeling unpredictability in polling results.

👉 [Open in Colab](https://drive.google.com/file/d/1-MYpXtK3qodGp9g-HHbXI73lmiiey7bw/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

[Back to Top](#-quick-navigation)

---

## 3. Normal Distribution

- Symmetric, bell-shaped curve defined by mean \( \mu \) and standard deviation \( \sigma \).
- Central to the Central Limit Theorem (CLT).

### Use Case Example
- **Healthcare**: Modeling blood pressure, IQ, cholesterol levels.
- **Real-time News**: Stock market returns approximation.

👉 [Open in Colab](https://drive.google.com/file/d/1-MYpXtK3qodGp9g-HHbXI73lmiiey7bw/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

[Back to Top](#-quick-navigation)

---

## 4. Sampling Distribution

- Distribution of a statistic (like the mean) over many samples from a population.
- Basis for estimating standard errors and confidence intervals.

### Use Case Example
- **Elections**: Estimating population vote share from poll samples.
- **Finance**: Estimating average portfolio return from multiple samples.

👉 [Open in Colab](https://drive.google.com/file/d/1-MYpXtK3qodGp9g-HHbXI73lmiiey7bw/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

[Back to Top](#-quick-navigation)

---

## 5. Interval Estimation

- Provides range of values for a population parameter (e.g., mean) using sample data.
- E.g., 95% Confidence Interval: \( \mu \in [\bar{x} \pm z \cdot \frac{\sigma}{\sqrt{n}}] \)

### Use Case Example
- **Healthcare**: Estimating average effectiveness of a new drug.
- **Economics**: Estimating unemployment rate ranges.

👉 [Open in Colab](https://drive.google.com/file/d/1-MYpXtK3qodGp9g-HHbXI73lmiiey7bw/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

[Back to Top](#-quick-navigation)

---

## 6. Hypothesis Testing

- Framework for making inferences about population parameters.
- Involves Null Hypothesis \( H_0 \), Alternative \( H_1 \), test statistic, p-value, and decision rule.

### Use Case Example
- **Medical Trials**: Testing drug effectiveness vs placebo.
- **A/B Testing**: Comparing two website versions for conversion.

👉 [Open in Colab](https://drive.google.com/file/d/1Wigcmj9mSDAsOCofSnT13A-0kgOFQNc3/view?usp=sharing)  
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

[Back to Top](#-quick-navigation)

---

## 7. CSV Download

👉 [Download CSV from Google Drive](https://drive.google.com/uc?export=download&id=1EujWEYaiq8UTBZLftnNlVFpFZXKoBqeZ)  
📎 [View CSV in Google Drive](https://drive.google.com/file/d/1EujWEYaiq8UTBZLftnNlVFpFZXKoBqeZ/view)

[Back to Top](#-quick-navigation)

---

## References & Further Reading

- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Seaborn Docs](https://seaborn.pydata.org/)
- [Plotly for Python](https://plotly.com/python/)
- [Khan Academy: Statistics](https://www.khanacademy.org/math/statistics-probability)
- [MIT OpenCourseWare](https://ocw.mit.edu/)
- [StatQuest with Josh Starmer (YouTube)](https://www.youtube.com/user/joshstarmer)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
---

**← Previous:** [Inferential Statistics](02-inferential-statistics.md)  
**→ Next:** [Hypothesis Testing - Part 1](04-hypothesis-testing-part-1.md)