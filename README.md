# Tri-SEM: A Robust Regression Framework
Zhilin Xiong, **Aihua Han**, Tiefeng Ma, Shuangzhe Liu

## Abstract
Outliers pose a major threat to the reliability of regression analysis. Unlike traditional robust methods that rely on numerical optimization, this paper presents Tri-SEM, a robust regression framework that leverages the morphological structure of data through a flexible three-stage design. In the first Split stage, data are partitioned into chain-like segments through the Anderson-Darling test, projection analysis, and convex hull detection to isolate potential outliers. In the second Extraction stage, a subset of clean segments is identified based on the size and median squared residuals. Finally, the Merge stage integrates all reliable inliers via a histogram transition detector analyzing residual distributions. Comprehensive experiments on diverse datasets demonstrate Tri-SEM's clear superiority in both prediction accuracy and estimation bias: it achieved the best overall rank and the highest prediction accuracy on 30 of the 35 datasets, while consistently outperforming the second-ranked method (MM-estimator) in estimation bias, achieving a relative improvement exceeding 90% on more than half (54.3%) of the datasets. Extensive ablation, sensitivity, convergence, and runtime analyses confirm the method’s robustness, efficiency, and adaptability across a wide range of data scenarios.


## File Structure
- Tri-SEM/
    - split.py
    - extraction.py
    - merge.py
    - trisem_main.py
    - datasets/
        - synthetic_data.py 
    - README.md

## Installation
The code is written in Python 3.8.8 and requires the following packages:

- `numpy`
- `scipy`
- `scikit-learn`


## Usage
```
from trisem_main import TriSEM
from datasets.synthetic_data import generate_scenario1

# Generate synthetic data
X, y, clean_location = generate_scenario1(n = 5000, p = 100,contam_rate=0.3)

# Initialize Tri-SEM
model = TriSEM(p_thr=0.5, c_thr=0.5, h_thr=0.5,max_iters=20, k=2)

# Fit model
model.fit(X, y)

# Access robust coefficients
beta_robust = model.beta_robust_
print("Robust coefficients:", beta_robust)
```
To generate other synthetic scenarios, call generate_scenario2(), generate_scenario3(), etc.


## Contact
If you have any questions, please contact aihuahan@zuel.edu.cn.
