# Convex Optimization and Support Vector Machines (SVM)

---

## Overview

This project explores the implementation of **Support Vector Machines (SVM)** using **convex optimization** principles and compares a **custom-built SVM (from scratch using CVXOPT)** with the **Scikit-Learn SVC**. The goal is to demonstrate how convex optimization ensures **global optimality**, how **kernel functions** enable non-linear classification, and how these concepts translate into high model accuracy in practice.

---

## 1. Conceptual Background

### 1.1 Convex Optimization in Machine Learning

Convex optimization is a branch of optimization where the objective function and constraints form a **convex set**.  
In such problems, any local minimum is also a **global minimum** — a property that makes it foundational to algorithms like SVM, logistic regression, and LASSO.

**General Form:**
```
minimize f(x)
subject to g_i(x) ≤ 0,   i = 1, 2, ..., m
            Ax = b
```

**Why it's powerful:**
- Guarantees **global solutions**
- Ensures **stability and robustness**
- Enables **efficient optimization** with standard solvers
- Reduces risk of overfitting via convex regularization terms

In the SVM case, convex optimization helps us find the hyperplane that **maximizes the margin** between two classes.

---

## 2. Support Vector Machines (SVM) Theory

### 2.1 What SVM Does

An SVM is a supervised machine learning algorithm used for **classification** and **regression** tasks.  
It separates data points into classes by finding the **optimal decision boundary (hyperplane)** that maximizes the distance (margin) between the nearest points from each class — called **support vectors**.

---

### 2.2 Hard Margin vs. Soft Margin

- **Hard Margin SVM:**  
  Assumes the data is perfectly linearly separable — no errors are allowed.
  
- **Soft Margin SVM:**  
  Introduces slack variables (ξ) to allow some misclassifications while still maximizing the margin.

**Soft Margin Objective:**
```
Minimize (1/2)||w||² + C * Σξ_i
Subject to: y_i(wᵀx_i + b) ≥ 1 - ξ_i,   ξ_i ≥ 0
```
Where:
- `w` = weight vector (defines the hyperplane)
- `b` = bias (offset)
- `C` = penalty parameter controlling the tradeoff between margin width and misclassification

---

### 2.3 The Dual Formulation and Kernel Trick

Solving the **dual form** of SVM is more efficient and allows for **non-linear classification** using kernels.

**Dual Problem:**
```
maximize  Σα_i - (1/2) ΣΣ α_i α_j y_i y_j K(x_i, x_j)
subject to Σα_i y_i = 0,   0 ≤ α_i ≤ C
```

The dual formulation replaces dot products with kernel functions, allowing the algorithm to **map data into higher dimensions** where it becomes linearly separable.

---

## 3. The Kernel Trick and the RBF Kernel

### 3.1 What is the Kernel Trick?

The kernel trick allows SVMs to handle non-linear data **without explicitly transforming it** into higher-dimensional space.  
It replaces the inner product `xᵀx'` with a **kernel function** `K(x, x')`, which measures similarity between data points.

---

### 3.2 RBF (Radial Basis Function) Kernel

**Formula:**
```
K(x, x') = exp(-γ * ||x - x'||²)
```
Where:
- `γ` controls the spread of the kernel (higher values → narrower influence)

**How it works:**
- The RBF kernel maps data into an **infinite-dimensional space**.
- It creates **circular decision boundaries**, ideal for datasets that are not linearly separable.
- In this project, it enables SVM to perfectly classify a **non-linear “two moons” dataset**.

---

## 4. Dataset Used

We used the **`make_moons` dataset** from Scikit-Learn, a common benchmark for testing non-linear classification models.

**Dataset Characteristics:**
- Two interleaving half-moon shapes.
- Binary classification (two classes: +1 and -1).
- Contains noise to simulate real-world imperfections.
- Perfect for testing the RBF kernel’s non-linear separation power.

---

## 5. Libraries Used

| Library | Purpose |
|----------|----------|
| **NumPy** | Numerical operations and array manipulation |
| **Matplotlib** | Visualization of decision boundaries and support vectors |
| **CVXOPT** | Solves quadratic convex optimization problems for custom SVM |
| **Scikit-Learn** | Provides built-in `SVC` for comparison |
| **StandardScaler** | Normalizes data to improve convergence |
| **make_moons** | Generates the non-linear dataset |

---

## 6. Implementation Summary

### 6.1 Custom SVM (From Scratch)

A custom **SVM classifier** was built using **CVXOPT**, which directly solves the **quadratic programming problem** in the dual form.

**Main Steps:**
1. Compute the **RBF kernel matrix** between all pairs of training points.  
2. Define the **quadratic optimization problem** matrices (P, q, G, h, A, b).  
3. Solve using CVXOPT’s `qp()` solver.  
4. Extract **support vectors** (where α_i > 0).  
5. Compute the **bias term (b)**.  
6. Make predictions based on the decision function:
   ```
   f(x) = Σ α_i y_i K(x, x_i) + b
   ```
7. Classify samples based on the sign of `f(x)`.

**Advantages:**
- Full control over optimization steps.
- Transparent understanding of dual formulation.
- Demonstrates convex optimization in action.

---

### 6.2 Scikit-Learn SVC

The Scikit-Learn implementation of SVM automates the same process internally using **LibSVM**, a highly optimized C-based solver.  
It efficiently handles kernel functions and regularization, providing a ready-to-use API.

**Steps performed internally:**
1. Computes kernel matrix.  
2. Solves the dual quadratic optimization problem.  
3. Determines support vectors and bias.  
4. Predicts with `sign(wᵀx + b)` in the transformed space.

**Advantages:**
- Highly optimized for performance.
- Robust with large datasets.
- Ideal for rapid prototyping and benchmarking.

---

## 7. Visualization and Decision Boundaries

Both models draw the **same decision boundary** because they solve the same convex optimization problem.  
The boundary separates the two moon-shaped classes with **maximum margin** and **minimal misclassification**.

- The **orange contour** represents the decision boundary (f(x) = 0).  
- The **magenta dashed lines** show margins (f(x) = ±1).  
- Support vectors are highlighted with black circles.

---

## 8. Experimental Results

### 8.1 Custom Kernel SVM (CVXOPT)

**Support Vectors (excerpt):**
```
[ 0.56949075 -0.73048257]
[ 1.28143258 -0.64417903]
[ 1.50760937  0.46871398]
...
[ 1.24875388 -0.33032216]
```

**Bias (b):** `-0.0538`

**Train Accuracy:** `99.18%`  
**Test Accuracy:** `99.52%`

---

### 8.2 Scikit-Learn SVM (SVC)

**Support Vectors Count:** `75`  
**Train Accuracy:** `99.18%`  
**Test Accuracy:** `99.52%`

---

### 8.3 Model Comparison

| Model               | Train Acc (%) | Test Acc (%) | # Support Vectors |
|---------------------|---------------|--------------|-------------------|
| Custom Kernel SVM   | 99.18         | 99.52        | 76                |
| Scikit-Learn SVC    | 99.18         | 99.52        | 75                |

---

## 9. Analysis and Insights

- **Accuracy:** Both implementations achieved near-perfect results, proving that the convex optimization solvers reached global optima.  
- **Support Vectors:** Almost identical between models; minor differences are due to numerical precision in solvers.  
- **RBF Kernel Effect:** Allowed SVM to correctly separate curved data boundaries that linear models would fail on.  
- **Convexity:** Ensured stable convergence and reproducible results — both implementations solved the same convex dual problem successfully.  
- **Performance:** Scikit-Learn’s optimized C backend is faster, but the CVXOPT implementation provides a transparent mathematical view.

---

## 10. Key Concepts Highlighted

| Concept | Explanation |
|----------|-------------|
| **Convex Optimization** | Guarantees global minimum, stable, and efficient optimization. |
| **Dual Formulation** | Converts the primal problem into a simpler, solvable quadratic form. |
| **Kernel Trick** | Allows non-linear classification without explicit feature transformation. |
| **RBF Kernel** | Transforms data into infinite dimensions to separate non-linear patterns. |
| **Support Vectors** | Points that lie on the margin boundaries and define the hyperplane. |
| **Slack Variables (ξ)** | Allow soft margins by tolerating small misclassifications. |
| **Penalty Parameter (C)** | Controls the balance between maximizing margin and minimizing error. |

---

## 11. Conclusion

This project demonstrates how **convex optimization** serves as the mathematical backbone of machine learning algorithms like SVM.  
By solving the **dual convex quadratic optimization problem**, SVMs guarantee **global optimal solutions** that generalize well to unseen data.

The **custom CVXOPT-based implementation** verifies the underlying mathematical process, while **Scikit-Learn’s SVC** offers a highly efficient practical tool.  
Both converge to the same results, validating that convex optimization ensures accuracy, stability, and optimal decision boundaries in AI models.

---
