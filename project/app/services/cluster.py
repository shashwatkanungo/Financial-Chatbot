import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_amount_vs_due(df: pd.DataFrame) -> pd.DataFrame:
    """Clustering with Amount vs Days_Until_Due."""
    features = ['Amount', 'Days_Until_Due']
    
    # Standardizing the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Plot clusters
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=df['Amount'], y=df['Days_Until_Due'], hue=df['Cluster'], palette='Set2')
    plt.title('Invoice Clusters based on Amount and Days_Until_Due')
    plt.savefig("outputs/eda/cluster_amount_vs_due.png")
    plt.close()

    return df

def cluster_amount_days_to_pay_due(df: pd.DataFrame) -> pd.DataFrame:
    """Clustering with Amount, Days_To_Pay, and Days_Until_Due."""
    features = ['Amount', 'Days_To_Pay', 'Days_Until_Due']
    
    # Standardizing the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster_Payment_Behavior'] = kmeans.fit_predict(X)

    # Plot 2D projection
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Days_To_Pay', y='Amount', hue='Cluster_Payment_Behavior', palette='Set1')
    plt.title('Clusters based on Amount, Days_To_Pay, Days_Until_Due')
    plt.savefig("outputs/eda/cluster_payment_behavior.png")
    plt.close()

    return df

def cluster_with_payment_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Clustering with Payment Terms."""
    df_encoded = df.copy()
    df_encoded['Payment_Terms_Code'] = df_encoded['Payment terms'].astype('category').cat.codes

    features = ['Amount', 'Days_To_Pay', 'Days_Until_Due', 'Payment_Terms_Code']
    
    # Standardizing the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_encoded[features])

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_encoded['Cluster_With_Terms'] = kmeans.fit_predict(X)

    # Plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df_encoded, x='Amount', y='Days_To_Pay', hue='Cluster_With_Terms', palette='Set2')
    plt.title('Clusters with Payment Terms Included')
    plt.savefig("outputs/eda/cluster_with_payment_terms.png")
    plt.close()

    return df_encoded

def cluster_with_early_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Clustering with Early Payment indicator."""
    df_encoded = df.copy()
    df_encoded['Early_Payment_Code'] = df_encoded['Early_Payment'].astype(int)  # Assuming it's boolean

    features = ['Amount', 'Days_To_Pay', 'Early_Payment_Code']
    
    # Standardizing the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_encoded[features])

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_encoded['Cluster_Early_Late'] = kmeans.fit_predict(X)

    # Plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df_encoded, x='Days_To_Pay', y='Amount', hue='Cluster_Early_Late', palette='Dark2')
    plt.title('Clusters based on Early Payment Behavior')
    plt.savefig("outputs/eda/cluster_early_late.png")
    plt.close()

    return df_encoded

def print_centroids(df: pd.DataFrame, kmeans) -> None:
    """Print centroids after clustering."""
    centroids = kmeans.cluster_centers_
    
    # Inverse scaling to get the centroids in original space
    scaler = StandardScaler()
    centroids = scaler.inverse_transform(centroids)

    for i, centroid in enumerate(centroids):
        print(f"Cluster {i}: Amount={centroid[0]:.2f}, Days_To_Pay={centroid[1]:.2f}, Early_Payment={centroid[2]:.2f}")

def perform_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform all clustering operations."""
    df = cluster_amount_vs_due(df)
    df = cluster_amount_days_to_pay_due(df)
    df = cluster_with_payment_terms(df)
    df = cluster_with_early_payment(df)

    return df
