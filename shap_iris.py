# ==========================================
# SHAP Explainable AI - Iris Classification
# ==========================================

# 1. Import libraries
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 2. Load Iris Dataset
# ==========================================

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

print("\nDataset loaded successfully!")
print("Dataset shape:", X.shape)

print("\nFirst 5 rows:")
print(X.head())

print("\nTarget classes:")
print(iris.target_names)


# ==========================================
# 3. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. Train Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ==========================================
# 5. Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. Check Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ==========================================
# 7. Create SHAP Explainer
# ==========================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print("SHAP values calculated successfully!")


# ==========================================
# 8. SHAP Summary Plots for Each Class
# ==========================================

# Class names
class_names = iris.target_names

# SHAP values for each class
for class_index, class_name in enumerate(class_names):

    print(f"\nGenerating SHAP plot for {class_name}...")

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values[:, :, class_index],
        X_test,
        feature_names=X_test.columns,
        show=False
    )

    plt.title(f"SHAP Feature Importance - {class_name}")

    plt.tight_layout()
    plt.show()


# ==========================================
# 9. Explain One Prediction
# ==========================================

sample_index = 0

sample = X_test.iloc[[sample_index]]

actual_class = y_test[sample_index]
predicted_class = model.predict(sample)[0]

print("\n========== SINGLE PREDICTION ==========")

print("Input flower:")
print(sample)

print(
    "\nActual class:",
    iris.target_names[actual_class]
)

print(
    "Predicted class:",
    iris.target_names[predicted_class]
)


# ==========================================
# 10. SHAP Explanation for One Prediction
# ==========================================

# ==========================================
# 10. Individual SHAP Explanation
# ==========================================

print("\nGenerating individual SHAP explanation...")

explainer_new = shap.TreeExplainer(model)

# Calculate SHAP values for the test data
shap_explanation = explainer_new.shap_values(
    X_test,
    check_additivity=False
)

# Get SHAP values for our selected sample
sample_shap = shap_explanation[sample_index]

# Create waterfall-style explanation using SHAP Explanation
explanation = shap.Explanation(
    values=sample_shap[:, predicted_class],
    base_values=explainer_new.expected_value[predicted_class],
    data=X_test.iloc[sample_index].values,
    feature_names=X_test.columns.tolist()
)

shap.plots.waterfall(
    explanation,
    max_display=10
)

plt.show()

plt.show()
# ==========================================
# 11. Test Your Own Flower
# ==========================================

print("\n================================")
print("     TEST YOUR OWN FLOWER")
print("================================")

sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))


# Create input DataFrame
user_input = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=iris.feature_names
)


# Make prediction
prediction = model.predict(user_input)[0]

prediction_probability = model.predict_proba(user_input)[0]


# Display result
print("\n================================")
print("           RESULT")
print("================================")

print("Your flower measurements:")
print(user_input)

print(
    "\nPredicted flower:",
    iris.target_names[prediction].upper()
)

print("\nPrediction probabilities:")

for name, probability in zip(
    iris.target_names,
    prediction_probability
):
    print(f"{name}: {probability:.2%}")