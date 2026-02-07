# EMPLOYEE-SEGMENTATION-USING-ML
Unsupervised ML project for employee segmentation using K-Means clustering, feature engineering, and PCA visualization to support people analytics and HR strategy insights.

👥 Employee Segmentation Using ML (KMeans Clustering)

This project applies unsupervised machine learning to segment employees into meaningful workforce groups using KMeans clustering and PCA visualization. The segmentation is based on income, tenure, satisfaction, experience, and work-life indicators to support people analytics and workforce profiling.

📌 Project Summary

Employee populations are not homogeneous — they differ across experience, tenure, satisfaction, and work-life balance. This project uses clustering techniques to identify natural employee segments using HR-related features.
The workflow includes feature engineering, scaling, cluster validation, clustering, profiling, and visualization.

🎯 Objective

--> Segment employees into meaningful groups
--> Use HR behavioral and tenure features for clustering
--> Validate optimal cluster count statistically
--> Create interpretable segment profiles
--> Visualize clusters using dimensionality reduction

🛠 Tools & Libraries

Python
Pandas
NumPy
Matplotlib
Seaborn

Methods used:
Feature Engineering
Standard Scaling
KMeans Clustering
Elbow Method
Silhouette Score
PCA Visualization

📊 Dataset
IBM HR Employee dataset containing employee attributes such as:
Age
Monthly Income
Years at Company
Total Working Years
Distance From Home
Job Satisfaction
nvironment Satisfaction
Work Life Balance
Training Times Last Year
Dataset used for learning and portfolio demonstration.

⚙️ Methodology
🔹 Feature Engineering
Income distribution was right-skewed. A log transformation was applied:
MonthlyIncome_log = log(1 + MonthlyIncome)
This reduces skewness and improves clustering stability.

Selected segmentation features:
Age
Log Income
Tenure
Experience
Commute distance
Satisfaction metrics
Work-life balance
Training frequency

🔹 Feature Scaling
All features were standardized using StandardScaler because KMeans is distance-based and sensitive to scale differences.

🔹 Optimal Cluster Selection
Two validation techniques were used:
Elbow Method
Plotted WCSS vs cluster count
Looked for diminishing improvement point
Silhouette Score
Measured cluster separation quality
Compared scores from K=2 to K=8
Based on validation, K = 3 clusters was selected.

🔹 KMeans Clustering
Final model:
n_clusters = 3
n_init = 20
random_state = 42
Each employee receives a cluster label representing their segment.

🔹 Cluster Profiling
Cluster-wise averages were computed for:
Age
Monthly Income
Years at Company
Total Working Years
Job Satisfaction
Work Life Balance
Environment Satisfaction
This produces interpretable employee segment profiles.

🔹 PCA Visualization
Principal Component Analysis (PCA) was used to reduce features into 2 dimensions for visualization.
A 2D scatter plot shows:
Cluster separation
Segment compactness
Visual validation of clustering

📈 Outputs Generated
The script produces:
Elbow curve plot
Silhouette score plot
Cluster profile summary table
PCA cluster scatter visualization
