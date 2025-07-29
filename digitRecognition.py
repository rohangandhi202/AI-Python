# Import libraries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import seaborn as sns

# Load data
digits = load_digits()
print(f"Dataset shape: {digits.data.shape}")
print(f"Number of classes: {len(np.unique(digits.target))}")
print(f"Feature range: {digits.data.min():.1f} to {digits.data.max():.1f}")

# Visualize multiple sample digits
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    ax = axes[i//5, i%5]
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.suptitle("Sample Digits from Dataset")
plt.tight_layout()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=42, stratify=digits.target
)

# Feature scaling (beneficial for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
predictions = {}

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

for name, model in models.items():
    # Use scaled data for distance-based algorithms
    if 'KNN' in name or 'SVM' in name or 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        # Cross-validation on scaled data
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # Cross-validation on original data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    accuracy = np.mean(preds == y_test)
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    predictions[name] = preds
    
    print(f"\n{name}:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Find best performing model
best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\n🏆 Best Model: {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})")

# Detailed classification report for best model
print(f"\n--- Classification Report for {best_model} ---")
print(classification_report(y_test, predictions[best_model]))

# Predictions comparison table
print("\n--- Predictions Comparison (First 15 samples) ---")
print("Index | Actual |", end="")
for model_name in models.keys():
    print(f" {model_name[:8]:>8} |", end="")
print()
print("-" * (10 + len(models) * 11))

for i in range(min(15, len(y_test))):
    print(f"{i:>5} | {y_test[i]:>6} |", end="")
    for model_name in models.keys():
        pred = predictions[model_name][i]
        # Highlight incorrect predictions
        if pred != y_test[i]:
            print(f" {pred:>8}*|", end="")
        else:
            print(f" {pred:>8} |", end="")
    print()

print("* indicates incorrect prediction")

# Visualize model comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
cv_scores = [results[name]['cv_mean'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.subplot(1, 2, 1)
bars = plt.bar(x, accuracies, color='skyblue', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.ylim(0.9, 1.0)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

plt.subplot(1, 2, 2)
bars = plt.bar(x, cv_scores, color='lightcoral', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('Cross-Validation Score')
plt.title('Cross-Validation Scores')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.ylim(0.9, 1.0)

# Add value labels on bars
for bar, cv in zip(bars, cv_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{cv:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Confusion matrices for top 2 models
top_2_models = sorted(results.keys(), key=lambda k: results[k]['accuracy'], reverse=True)[:2]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = ['Blues', 'Greens']

for i, model_name in enumerate(top_2_models):
    cm = confusion_matrix(y_test, predictions[model_name])
    disp = ConfusionMatrixDisplay(cm, display_labels=digits.target_names)
    disp.plot(ax=axes[i], cmap=colors[i], values_format='d')
    axes[i].set_title(f"Confusion Matrix: {model_name}")

plt.tight_layout()
plt.show()

# Analyze misclassifications for best model
print(f"\n--- Misclassification Analysis for {best_model} ---")
misclassified = y_test != predictions[best_model]
misclassified_indices = np.where(misclassified)[0]

if len(misclassified_indices) > 0:
    print(f"Total misclassifications: {len(misclassified_indices)}")
    
    # Show some misclassified examples
    n_examples = min(6, len(misclassified_indices))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(n_examples):
        idx = misclassified_indices[i]
        test_idx = idx  # Index in test set
        
        # Find the original image index
        # This is approximate since we can't easily map back to original indices
        axes[i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
        axes[i].set_title(f"True: {y_test[idx]}, Pred: {predictions[best_model][idx]}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_examples, 6):
        axes[i].axis('off')
    
    plt.suptitle(f"Misclassified Examples - {best_model}")
    plt.tight_layout()
    plt.show()
else:
    print("Perfect classification! No misclassifications found.")

# Feature importance for Random Forest (if it's in top models)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_.reshape(8, 8)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(importances, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Importance Heatmap (Random Forest)')
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)