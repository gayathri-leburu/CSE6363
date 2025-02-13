import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Step 1: Load the USPS dataset

def load_dataset():
    usps = fetch_openml('usps', version=2)  # Fetch a newer version of the dataset
    X = usps.data.apply(pd.to_numeric, errors='coerce').astype(float)  # Ensure X is float
    y = usps.target.astype(int)  # Convert labels to integer
    X /= 255.0  # Normalize pixel values (0-255) to (0-1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_dataset()
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Step 2: Implement KNN Algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in np.array(X_test)]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return manhattan_distance(x1, x2)
        else:
            raise ValueError("Invalid distance metric")

# Step 3: Train and Evaluate Model
knn = KNNClassifier(k=3, distance_metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Compute performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
print("Classification Report:")
print(class_report)

# Step 4: Experiment with Different K Values
k_values = [1, 3, 5, 7, 9]
k_accuracies = []

for k in k_values:
    knn = KNNClassifier(k=k, distance_metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    k_accuracies.append(acc)
    print(f"K={k}, Accuracy: {acc:.4f}")

# Plot accuracy vs. K
plt.figure(figsize=(8, 5))
plt.plot(k_values, k_accuracies, marker='o', linestyle='-')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. K')
plt.grid(True)
plt.show()

# Step 5: Compare Euclidean vs. Manhattan Distance
metrics = ['euclidean', 'manhattan']
results = {}

for metric in metrics:
    knn = KNNClassifier(k=3, distance_metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[metric] = acc
    print(f"Metric: {metric}, Accuracy: {acc:.4f}")

# Plot Comparison
plt.figure(figsize=(6, 4))
plt.bar(results.keys(), results.values(), color=['blue', 'red'])
plt.xlabel('Distance Metric')
plt.ylabel('Accuracy')
plt.title('Comparison of Distance Metrics')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Step 6: Analyze Misclassified Examples
def analyze_misclassifications(y_test, y_pred, X_test):
    misclassified_indices = np.where(y_test != y_pred)[0]
    print(f"Total misclassified examples: {len(misclassified_indices)}")
    
    # Plot some misclassified images
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_indices):
            idx = misclassified_indices[i]
            ax.imshow(X_test[idx].reshape(16, 16), cmap='gray')
            ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
            ax.axis('off')
    plt.show()

# Evaluate and Analyze Misclassifications
analyze_misclassifications(y_test, y_pred, X_test)
