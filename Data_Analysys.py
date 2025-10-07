
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
file_name = "wustl-ehms-2020_with_attacks_categories.csv"  # Replace with your dataset name
dataset = pd.read_csv(file_name)

# Step 2: Drop unnecessary columns
columns_to_drop = ["SrcAddr", "DstAddr", "SrcMac", "DstMac", "Dir", "Flgs"]
processed_dataset = dataset.drop(columns=columns_to_drop, axis=1)

# Step 3: Convert categorical columns to numerical using Label Encoding
label_encoder_sport = LabelEncoder()
processed_dataset["Sport"] = label_encoder_sport.fit_transform(processed_dataset["Sport"])

label_encoder_attack_category = LabelEncoder()
processed_dataset["Attack Category"] = label_encoder_attack_category.fit_transform(processed_dataset["Attack Category"])

# Step 4: Remove outliers only from the majority class
numeric_columns = processed_dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('Label')  # Exclude the target label column

# Separate the majority and minority classes
class_0 = processed_dataset[processed_dataset['Label'] == 0]  # Majority class
class_1 = processed_dataset[processed_dataset['Label'] == 1]  # Minority class

# Apply IQR outlier removal only on the majority class
for column in numeric_columns:
    Q1 = class_0[column].quantile(0.25)
    Q3 = class_0[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    class_0 = class_0[(class_0[column] >= lower_bound) & (class_0[column] <= upper_bound)]

# Combine the cleaned majority class with the minority class
data_clean = pd.concat([class_0, class_1], axis=0)

# Step 5: Split the data into training and testing sets
X = data_clean.drop('Label', axis=1)  # Features
y = data_clean['Label']  # Target

print("Class distribution in full dataset:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Check class distribution in y_train
print("Class distribution in y_train before SMOTE:\n", y_train.value_counts())

# Step 6: Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# Step 7: Standardize the numerical features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)  # Standardize training data
X_test = scaler.transform(X_test)  # Standardize test data

# Final Summary
print("Original dataset shape:", processed_dataset.shape)
print("Cleaned dataset shape:", data_clean.shape)
print("Training set shape after resampling:", X_train_resampled.shape)
print("Testing set shape:", X_test.shape)



import matplotlib.pyplot as plt
import seaborn as sns

# Set general Seaborn style
sns.set(style="whitegrid")

# Step 1: Plot histograms to understand feature distributions
numeric_columns = data_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('Label')  # Exclude the target variable

plt.figure(figsize=(15, 12))
for i, column in enumerate(numeric_columns[:12]):  # Display first 12 columns for clarity
    plt.subplot(4, 3, i + 1)
    sns.histplot(data_clean[column], kde=True, color='skyblue')
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
plt.show()

# Step 2: Plot boxplots to identify outliers
plt.figure(figsize=(15, 12))
for i, column in enumerate(numeric_columns[:12]):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(x=data_clean[column], color='lightcoral')
    plt.title(f"Boxplot of {column}")
    plt.tight_layout()
plt.show()

# Step 3: Visualize correlations using a Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data_clean[numeric_columns + ['Label']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Scatter plot to examine relationships between key features
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data_clean, x='SrcBytes', y='DstBytes', hue='Label', palette='viridis')
plt.title("Scatter Plot between SrcBytes and DstBytes")
plt.xlabel("Source Bytes")
plt.ylabel("Destination Bytes")
plt.show()





import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Step 1: Plot histograms for feature distributions
numeric_columns = data_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('Label')  # Exclude target variable

plt.figure(figsize=(15, 12))
for i, column in enumerate(numeric_columns[:12]):  # Display first 12 features for clarity
    plt.subplot(4, 3, i + 1)
    sns.histplot(data_clean[column], kde=True, color='skyblue')
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
plt.show()

# Step 2: Plot boxplots to detect outliers
plt.figure(figsize=(15, 12))
for i, column in enumerate(numeric_columns[:12]):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(x=data_clean[column], color='lightcoral')
    plt.title(f"Boxplot of {column}")
    plt.tight_layout()
plt.show()

# Step 3: Heatmap for correlation analysis
plt.figure(figsize=(15, 12))
correlation_matrix = data_clean[numeric_columns + ['Label']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()

# Step 4: Scatter plot for relationships between selected features
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data_clean, x='SrcBytes', y='DstBytes', hue='Label', palette='viridis')
plt.title("Scatter Plot: SrcBytes vs DstBytes")
plt.xlabel("Source Bytes")
plt.ylabel("Destination Bytes")
plt.show()



# Import necessary libraries
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Split the data into training and testing sets (70% train, 30% test)
X = data_clean.drop('Label', axis=1)  # Features
y = data_clean['Label']  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Step 2: Define the parameter grid for NearestCentroid
param_grid_nc = {
    'metric': ['euclidean', 'manhattan', 'cosine'],  # Different distance metrics
    'shrink_threshold': [None, 0.1, 0.5, 1.0]       # Shrink threshold values
}

# Step 3: Initialize the NearestCentroid model
nc_model = NearestCentroid()

# Step 4: Perform Grid Search with Cross Validation
grid_search_nc = GridSearchCV(estimator=nc_model, param_grid=param_grid_nc,
                              cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Step 5: Fit the model to the training data
grid_search_nc.fit(X_train, y_train)

# Step 6: Get the best parameters and evaluate the model
best_nc = grid_search_nc.best_estimator_
print("Best Parameters for NearestCentroid:", grid_search_nc.best_params_)

# Step 7: Predict and evaluate on the test set
y_pred_nc = best_nc.predict(X_test)
print("\nAccuracy on Test Set:", accuracy_score(y_test, y_pred_nc))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nc))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nc))

