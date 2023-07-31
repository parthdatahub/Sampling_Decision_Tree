# Random Sampling and Decision Tree:

## Random Sampling:

Random sampling involves selecting data points from a population randomly. Each data point has an equal chance of being selected. Random sampling is simple and unbiased.

Example (Python code using NumPy):
```python
import numpy as np

# Sample data
population = np.arange(1, 101)

# Randomly select 10 data points
sample_size = 10
random_sample = np.random.choice(population, size=sample_size, replace=False)

# Oversampling and Undersampling:
Oversampling and undersampling are techniques used to handle imbalanced datasets, where one class is underrepresented compared to others.

Oversampling: It involves increasing the instances of the minority class to balance the dataset.
Undersampling: It involves reducing the instances of the majority class to balance the dataset.
Example (Python code using imbalanced-learn):

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Sample data with features (X) and target labels (y) - imbalanced dataset
X = np.random.randn(100, 5)
y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

# Oversampling
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Undersampling
undersampler = RandomUnderSampler()
X_resampled, y_resampled = undersampler.fit_resample(X, y)

Decision Tree:
Decision tree is a popular machine learning algorithm used for both classification and regression tasks. It recursively partitions the data into subsets based on features, resulting in a tree-like structure. Decision trees are interpretable and can handle both numerical and categorical data.

How Decision Trees Work:

Splitting:
The decision tree algorithm finds the best feature and split point to partition the data based on a criterion (e.g., Gini impurity for classification or mean squared error for regression).

Recursive Process:
The splitting process is repeated on each subset until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf, or impurity threshold).

Leaf Nodes:
The final nodes of the decision tree are called leaf nodes and represent the predicted class (classification) or value (regression) for new data points.

