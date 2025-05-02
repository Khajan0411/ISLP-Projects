Okay, here are the simplified notes for Chapter 4: Classification.

## Chapter 4: Classification - Simplified Notes

This chapter focuses on methods for predicting a qualitative (categorical) response variable. This task is known as classification.

**1. Overview of Classification**

* Unlike regression where the output is numeric, classification predicts which category or class an observation belongs to.
* Examples: Predicting medical conditions (stroke, overdose, seizure), identifying fraudulent transactions (yes/no), determining DNA mutation types.

**2. Why Not Linear Regression?**

* If the qualitative response has more than two categories (e.g., stroke, overdose, seizure), coding them numerically (like 1, 2, 3) imposes an artificial order and distance between categories that usually doesn't make sense. Different codings lead to different models and predictions.
* If the response has only two categories (binary, e.g., Yes/No, coded as 0/1), linear regression *can* be used. The prediction $\hat{Y}$ can be interpreted as an estimate of the probability $Pr(Y=1|X)$. However, the predicted probabilities can fall outside the sensible range of [0, 1].
* Therefore, methods designed specifically for classification are preferred.

**3. Logistic Regression**

* Models the *probability* that the response $Y$ belongs to a particular category, given the predictors $X$.
* **Binary Case (2 Classes):** Models $Pr(Y=1|X) = p(X)$.
    * Uses the *logistic function* to ensure probabilities are between 0 and 1:
        $p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + ... + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + ... + \beta_p X_p}}$
    * This model implies a linear relationship for the *log-odds* (or *logit*):
        $\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p$
    * Interpretation: A one-unit increase in $X_j$ changes the log-odds by $\beta_j$, or multiplies the odds by $e^{\beta_j}$. The change in *probability* $p(X)$ depends on the current value of $X$.
* **Estimating Coefficients:** Uses *maximum likelihood* to find coefficients ($\hat{\beta}_0, \hat{\beta}_1, ...$) that make the predicted probabilities closest to the actual observed class labels (0 or 1) in the training data.
* **Making Predictions:** Calculate $\hat{p}(X)$ using the estimated coefficients. Classify as '1' if $\hat{p}(X)$ > threshold (often 0.5), and '0' otherwise.
* **Multiple Logistic Regression:** Uses multiple predictors ($X_1, ..., X_p$). Interpretation needs care due to potential *confounding* (correlation between predictors can mask or distort the relationship of a single predictor with the response).
* **Multinomial Logistic Regression:** Extends logistic regression to handle responses with more than two classes ($K > 2$). One class is chosen as baseline, and the model estimates log-odds relative to that baseline. An alternative is the *softmax* coding which treats classes symmetrically.

**4. Generative Models for Classification**

* Alternative approach: Instead of directly modeling $Pr(Y=k|X)$, model the distribution of predictors $X$ *within each class*, $f_k(X) = Pr(X|Y=k)$.
* Also estimate the *prior probability* $\pi_k = Pr(Y=k)$ (overall probability of belonging to class k).
* Use *Bayes' Theorem* to combine these for the posterior probability:
    $Pr(Y=k|X=x) = \frac{\pi_k f_k(x)}{\sum_{l=1}^{K} \pi_l f_l(x)}$
* Classify to the class with the highest posterior probability (this approximates the *Bayes Classifier*, which has the lowest possible error rate).
* **Linear Discriminant Analysis (LDA):**
    * Assumes $f_k(x)$ is a multivariate Gaussian (normal) distribution for each class $k$.
    * Crucially, assumes all classes share the *same* covariance matrix ($\Sigma$).
    * Results in *linear* decision boundaries between classes.
* **Quadratic Discriminant Analysis (QDA):**
    * Also assumes $f_k(x)$ is multivariate Gaussian for each class $k$.
    * Allows each class to have its *own* covariance matrix ($\Sigma_k$).
    * Results in *quadratic* decision boundaries.
    * More flexible than LDA but requires estimating more parameters, so needs more data (higher variance).
* **LDA vs. QDA:** LDA is better if the shared covariance assumption holds or if $n$ is small relative to $p$ (lower variance). QDA is better if the shared covariance assumption is poor or if $n$ is large (bias is lower).
* **Naive Bayes:**
    * Does *not* assume a specific distribution family (like Gaussian) for $f_k(x)$.
    * Instead, makes a strong assumption: within each class $k$, the predictors $X_1, ..., X_p$ are *independent*.
    * This simplifies estimation greatly: only need to estimate the one-dimensional density $f_{kj}(x_j)$ for each predictor $j$ within each class $k$, rather than the full p-dimensional density $f_k(x)$.
    * The independence assumption is often violated ("naive"), but the method performs surprisingly well, especially when $p$ is large relative to $n$, due to variance reduction.

**5. Comparison of Classification Methods**

* **LDA, Logistic Regression:** Assume linear decision boundaries. LDA performs better if Gaussian assumption holds; logistic regression is more robust if not.
* **QDA:** Assumes quadratic boundaries (more flexible than linear). Better if true boundary is non-linear and $n$ is large enough.
* **Naive Bayes:** Can model non-linear boundaries via its additive structure on the log-odds scale. Often good when $p$ is large. Cannot inherently model interactions between predictors like QDA can.
* **KNN:** Non-parametric. Can model highly non-linear boundaries. Needs $n$ to be much larger than $p$ to perform well (curse of dimensionality). Less interpretable.
* **No Free Lunch:** No single method dominates. Choice depends on the data (linearity of boundary, $n$ vs $p$, distribution assumptions) and goals (prediction vs. inference). Cross-validation (Chapter 5) helps choose.

**6. Generalized Linear Models (GLMs)**

* A broader framework encompassing linear regression, logistic regression, and Poisson regression (for count data).
* **Common Characteristics:**
    1.  Assume the response $Y$ follows a distribution from the *exponential family* (e.g., Gaussian for linear regression, Bernoulli for logistic, Poisson for counts).
    2.  Model the mean of $Y$ using a *link function* $\eta$ such that the transformed mean is a linear function of predictors: $\eta(E(Y|X_1,...,X_p)) = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p$.
* **Poisson Regression Example (Bikeshare):**
    * Response $Y$ (number of bikers) is a non-negative integer (count).
    * Assumes $Y$ follows a Poisson distribution with mean $\lambda$.
    * Models $\log(\lambda(X)) = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p$. Ensures predicted mean ($\lambda$) is non-negative.
    * Handles the mean-variance relationship often seen in count data (variance tends to increase with the mean), unlike standard linear regression which assumes constant variance.

Let me know if you would like the notes for Chapter 5!
