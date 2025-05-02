### Chapter 3: Linear Regression - Simplified Notes

This chapter focuses on linear regression, a fundamental method for predicting a quantitative (numeric) response variable based on one or more predictor variables.

**1. Simple Linear Regression (One Predictor)**

* **Concept:** Assumes an approximately linear relationship between a single predictor $X$ and the response $Y$.
* **Model:** $Y \approx \beta_0 + \beta_1 X$
    * $\beta_0$ is the *intercept* (predicted value of $Y$ when $X=0$).
    * $\beta_1$ is the *slope* (average change in $Y$ for a one-unit increase in $X$).
    * $\beta_0$ and $\beta_1$ are unknown *coefficients* or *parameters*.
* **Estimating Coefficients:** We estimate $\beta_0$ and $\beta_1$ using the training data to find the line that is "closest" to the data points. The most common method is *least squares*.
    * **Least Squares:** Chooses coefficient estimates ($\hat{\beta}_0, \hat{\beta}_1$) that minimize the *Residual Sum of Squares (RSS)*.
    * **Residual:** The difference between the actual response ($y_i$) and the predicted response ($\hat{y}_i$) for the $i$-th observation ($e_i = y_i - \hat{y}_i$).
    * **RSS:** Sum of the squared residuals for all observations ($RSS = \sum e_i^2$).
* **Assessing Accuracy of Coefficients:**
    * **Standard Error (SE):** Measures how much the estimated coefficients ($\hat{\beta}_0, \hat{\beta}_1$) would vary if we used a different training dataset. Lower SE means more confidence in the estimate.
    * **Confidence Intervals:** Provide a range of plausible values for the true coefficients ($\beta_0, \beta_1$). E.g., a 95% confidence interval means there's a 95% probability that the range contains the true value.
    * **Hypothesis Testing:** Most commonly used to test $H_0: \beta_1 = 0$ (no relationship between X and Y) vs. $H_a: \beta_1 \neq 0$ (there is a relationship).
        * Uses a *t-statistic*: measures how many standard deviations $\hat{\beta}_1$ is from 0.
        * Uses a *p-value*: the probability of observing such an extreme t-statistic if $H_0$ were true. Small p-value (e.g., < 0.05) suggests rejecting $H_0$.
* **Assessing Model Accuracy:**
    * **Residual Standard Error (RSE):** An estimate of the standard deviation of the error term ($\epsilon$). Represents the average amount the actual responses deviate from the true regression line. Measured in units of Y.
    * **R² Statistic:** Measures the *proportion of variance explained* by the model. Always between 0 and 1. Higher R² means the model explains more variability in Y. R² = 1 - RSS/TSS (where TSS is Total Sum of Squares, measuring total variance in Y). In simple linear regression, R² = (correlation between X and Y)².

**2. Multiple Linear Regression (Multiple Predictors)**

* **Concept:** Extends simple linear regression to use multiple predictors ($X_1, ..., X_p$).
* **Model:** $Y \approx \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p$
* **Interpretation:** $\beta_j$ is the average effect on $Y$ of a one-unit increase in $X_j$, *holding all other predictors fixed*.
* **Estimating Coefficients:** Still uses least squares to minimize RSS. Formulas are more complex (often uses matrix algebra).
* **Important Questions:**
    1.  **Is there a relationship?** Test $H_0: \beta_1 = ... = \beta_p = 0$ using the *F-statistic*. A large F-statistic (and small p-value) indicates at least one predictor is related to $Y$.
    2.  **Which predictors are important?** This is *variable selection*. Looking at individual p-values for each $\beta_j$ can be misleading if p is large. Formal methods include:
        * **Forward Selection:** Start with no predictors, add them one-by-one based on best improvement in fit.
        * **Backward Selection:** Start with all predictors, remove the least useful one-by-one.
        * **Mixed Selection:** Combination of forward and backward.
    3.  **How well does the model fit?** Use RSE and R² (now R² = Correlation(Y, $\hat{Y}$)²). Adding variables *always* increases R², but not necessarily *Adjusted R²*, which penalizes for model complexity.
    4.  **How accurate are predictions?** Use *confidence intervals* for the average response and *prediction intervals* for individual responses. Prediction intervals are wider because they include irreducible error ($\epsilon$).

**3. Other Considerations**

* **Qualitative Predictors:**
    * Use *dummy variables*. If a predictor has K levels, create K-1 dummy variables. The level without a dummy variable is the *baseline*.
    * The coefficients represent differences relative to the baseline level.
* **Extensions of the Linear Model:**
    * **Interaction Terms (Synergy):** Allows the effect of one predictor to depend on the value of another. Add product terms like $X_1 \times X_2$ to the model ($Y \approx \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2$). The *hierarchical principle* suggests keeping main effects ($X_1, X_2$) if their interaction is included.
    * **Non-linear Relationships:** Model non-linearity using transformations like $\log(X), \sqrt{X}, X^2, X^3$ (*polynomial regression*). This is still a linear model in terms of the *transformed* predictors.
* **Potential Problems:**
    1.  **Non-linearity:** If the true relationship isn't linear, model conclusions are suspect. Check *residual plots* (residuals vs. fitted values) for patterns. Fix by transforming predictors (e.g., polynomials) or using methods from Chapter 7.
    2.  **Correlation of Error Terms:** Assumes errors $\epsilon_i$ are uncorrelated. Common in *time series*. Can lead to underestimated standard errors (overly narrow intervals, too small p-values). Check by plotting residuals over time/sequence.
    3.  **Non-constant Variance of Errors (Heteroscedasticity):** Assumes error variance is constant. Look for funnel shapes in residual plots. Fix by transforming Y (e.g., $\log(Y), \sqrt{Y}$) or using *weighted least squares*.
    4.  **Outliers:** Observations with unusual Y values given X. Can inflate RSE and affect p-values, but may not drastically change the line if leverage is low. Identify using *studentized residuals* (residuals divided by their estimated SE).
    5.  **High Leverage Points:** Observations with unusual X values. Can have a large impact on the fitted line. Identify using the *leverage statistic*. Outliers with high leverage are particularly dangerous.
    6.  **Collinearity:** Two or more predictors are closely related. Makes it hard to separate their individual effects. Increases standard errors of coefficients, reducing power to detect effects (larger p-values). Check correlation matrix or *Variance Inflation Factor (VIF)*. Fix by dropping a variable or combining collinear variables.

**4. Comparison with K-Nearest Neighbors (KNN)**

* **Linear Regression:** Parametric (assumes linear form). Simple, interpretable. Performs well if the true relationship is linear. Can perform poorly if relationship is non-linear. Performs better than KNN when $p$ is large relative to $n$.
* **KNN Regression:** Non-parametric (makes no assumption about the form of $f$). More flexible. Can outperform linear regression if the relationship is non-linear, *provided $n$ is large and $p$ is small*. Suffers from the *curse of dimensionality* (performance degrades as $p$ increases because nearest neighbors become far away). Less interpretable.

