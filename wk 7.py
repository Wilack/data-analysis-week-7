# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for better looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the Iris dataset
try:
    iris = load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                          columns=iris['feature_names'] + ['target'])
    
    # Map target values to species names
    iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {iris_df.shape}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
iris_df.head()

# Explore the structure of the dataset
print("\nData Types:")
print(iris_df.dtypes)

print("\nMissing Values:")
print(iris_df.isnull().sum())

# Clean the dataset (though Iris dataset is already clean)
# For demonstration, we'll show how we would handle missing values if there were any
if iris_df.isnull().sum().sum() > 0:
    print("\nCleaning dataset...")
    # Fill numerical missing values with mean
    num_cols = iris_df.select_dtypes(include=['float64']).columns
    iris_df[num_cols] = iris_df[num_cols].fillna(iris_df[num_cols].mean())
    
    # Fill categorical missing values with mode
    cat_cols = iris_df.select_dtypes(include=['object']).columns
    iris_df[cat_cols] = iris_df[cat_cols].fillna(iris_df[cat_cols].mode().iloc[0])
else:
    print("\nNo missing values found. Dataset is clean.")

# Compute basic statistics for numerical columns
print("Basic Statistics:")
print(iris_df.describe())

# Group by species and compute mean of numerical columns
print("\nMean by Species:")
species_stats = iris_df.groupby('species').mean()
print(species_stats)

# Interesting findings observation
print("\nObservations:")
print("- Setosa has significantly smaller petal length and width compared to other species")
print("- Virginica has the largest sepal length on average")
print("- Versicolor is intermediate in most measurements between setosa and virginica")

plt.figure(figsize=(12, 6))
iris_df.sort_values('sepal length (cm)').reset_index()['sepal length (cm)'].plot(
    color='green', linewidth=2)
plt.title('Trend of Sepal Length Across Observations (Sorted)')
plt.xlabel('Observation Index (Sorted by Sepal Length)')
plt.ylabel('Sepal Length (cm)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
species_stats[['sepal length (cm)', 'sepal width (cm)', 
               'petal length (cm)', 'petal width (cm)']].plot(kind='bar')
plt.title('Average Measurements by Iris Species')
plt.ylabel('Centimeters (cm)')
plt.xlabel('Species')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', 
                hue='species', style='species', s=100, palette='dark')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# Boxplot to show distribution and outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=iris_df.drop('target', axis=1), palette='Set2')
plt.title('Distribution of Measurements Across All Species')
plt.ylabel('Centimeters (cm)')
plt.xticks(rotation=45)
plt.show()


# Pairplot to show all relationships
sns.pairplot(iris_df.drop('target', axis=1), hue='species', palette='husl', height=2.5)
plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=iris_df, x='petal length (cm)', hue='species', 
             element='step', kde=True, palette='viridis')
plt.title('Distribution of Petal Length by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()
