import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Madhurima\Desktop\python\practice_madhurimaa\ml_hr_attrition_prediction\Employee_Attrition.csv")

# FEATURE ENGINEERING

# Handle skewed income
df['MonthlyIncome_log'] = np.log1p(df['MonthlyIncome'])

# Select richer HR segmentation features
features = df[[
    'Age',
    'MonthlyIncome_log',
    'YearsAtCompany',
    'TotalWorkingYears',
    'DistanceFromHome',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance',
    'TrainingTimesLastYear'
]]

# SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ELBOW METHOD
from sklearn.cluster import KMeans

wcss = []

for k in range(1, 9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,9), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# SILHOUETTE VALIDATION
from sklearn.metrics import silhouette_score

sil_scores = []

for k in range(2,9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    labels = kmeans.fit_predict(scaled_features)
    sil_scores.append(silhouette_score(scaled_features, labels))

plt.plot(range(2,9), sil_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# FINAL MODEL (choose best K after checking silhouette)
best_k = 3

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

df['Cluster'] = kmeans.fit_predict(scaled_features)

# CLUSTER PROFILE TABLE
cluster_profile = df.groupby('Cluster')[[
    'Age',
    'MonthlyIncome',
    'YearsAtCompany',
    'TotalWorkingYears',
    'JobSatisfaction',
    'WorkLifeBalance',
    'EnvironmentSatisfaction'
]].mean().round(2)

print("\nCluster Profile Summary:")
print(cluster_profile)

# PCA VISUALIZATION
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

df['PCA1'] = pca_features[:,0]
df['PCA2'] = pca_features[:,1]

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    palette='Set2',
    alpha=0.7
)

plt.title("Employee Segments — KMeans Clustering (PCA View)")
plt.show()
