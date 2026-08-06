# Customer Segmentation using Unsupervised Learning

## Project Overview

This project performs customer segmentation using unsupervised machine learning algorithms. It groups customers based on their purchasing behavior using K-Means and DBSCAN clustering algorithms.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly

## Dataset

Mall Customers Dataset

Features Used:
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Algorithms

### K-Means
- Number of Clusters: 5

### DBSCAN
- eps = 0.8
- min_samples = 5

## Visualization

Interactive 3D Scatter Plot using Plotly.

## Output

- Customer clusters using K-Means
- Customer clusters using DBSCAN
- Clustered dataset saved as `clustered_customers.csv`

## How to Run

```bash
pip install -r requirements.txt
python app.py
```