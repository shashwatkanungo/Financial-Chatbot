import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway, kruskal
import math

# Optional: Directory for saving plots
PLOT_DIR = "outputs/eda"
os.makedirs(PLOT_DIR, exist_ok=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean out-of-range float values (infinity and NaN) in the DataFrame.
    """
    # Replace infinity values with None (or NaN)
    df.replace([float('inf'), float('-inf')], None, inplace=True)
    
    # Replace NaN values with a default value (e.g., 0)
    df.fillna(0, inplace=True)
    
    return df

def safe_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    return obj


def analyze_data(df: pd.DataFrame) -> dict:
    df = clean_data(df)
    results = perform_eda(df)
    return safe_json(results)


def perform_eda(df: pd.DataFrame) -> dict:
    results = {}

    # Descriptive statistics
    results["summary_stats"] = df[['Amount', 'Days_To_Pay', 'Days_Until_Due']].describe().to_dict()
    results["early_payment_ratio"] = df['Early_Payment'].value_counts(normalize=True).to_dict()
    results["payment_terms_count"] = df['Payment terms'].value_counts().to_dict()

    # Plot: Distribution of Days to Pay
    plot_hist(df, "Days_To_Pay", "Distribution of Days to Pay")

    # Plot: Distribution of Invoice Amount
    plot_hist(df, "Amount", "Distribution of Invoice Amount")

    # Boxplot: Amount vs Early Payment
    plot_box(df, "Early_Payment", "Amount", "Invoice Amount vs Early Payment")

    # Boxplot: Days_To_Pay by Country (top 10)
    top_countries = df['Country'].value_counts().head(10).index
    plot_box(df[df['Country'].isin(top_countries)], "Country", "Days_To_Pay", "Days to Pay by Country (Top 10)")

    # Time-based trends
    df['Invoice Month'] = pd.to_datetime(df['Invoice date']).dt.to_period('M').astype(str)
    monthly_avg = df.groupby('Invoice Month')['Days_To_Pay'].mean().reset_index()
    plot_line(monthly_avg, 'Invoice Month', 'Days_To_Pay', 'Average Days to Pay Over Time')

    # Revenue type vs days to pay
    plot_box(df, 'Revenue Type', 'Days_To_Pay', 'Days to Pay by Revenue Type')

    # Payment method vs days to pay
    plot_bar(df, 'Payment method', 'Days_To_Pay', 'Days to Pay by Payment Method')

    # Credit card trend over years
    df['Year'] = pd.to_datetime(df['Invoice date']).dt.year
    cc_df = df[df['Payment method'] == 'Credit card']
    cc_trend = cc_df.groupby('Year')['Days_To_Pay'].mean().reset_index()
    plot_line(cc_trend, 'Year', 'Days_To_Pay', 'Avg Days to Pay for Credit Card Payments')

    # Credit card trend monthly
    cc_monthly = cc_df.groupby('Invoice Month')['Days_To_Pay'].mean().reset_index()
    plot_line(cc_monthly, 'Invoice Month', 'Days_To_Pay', 'Monthly Trend: Credit Card Payments')

    # Payment terms analysis
    plot_box(df.head(1000), 'Payment terms', 'Days_To_Pay', 'Days to Pay by Payment Terms')

    # Summary stats + ANOVA/Kruskal
    stats = df.groupby('Payment terms')['Days_To_Pay'].agg(['count', 'mean', 'median', 'std', 'var']).reset_index()
    results["payment_terms_stats"] = stats.to_dict()

    groups = [group['Days_To_Pay'].dropna() for _, group in df.groupby('Payment terms')]
    results["anova_p"] = f_oneway(*groups).pvalue
    results["kruskal_p"] = kruskal(*groups).pvalue

    # Violin plot
    plot_violin(df.head(1000), 'Payment terms', 'Days_To_Pay', 'Variability in Days to Pay')

    # Scatter plot and correlation
    plot_scatter(df, 'Days_Until_Due', 'Days_To_Pay', 'Due vs Days to Pay')
    results["due_pay_correlation"] = df[['Days_Until_Due', 'Days_To_Pay']].corr().iloc[0, 1]

    # Billing buckets
    df_clean = df[['Days_Until_Due', 'Days_To_Pay']].dropna()
    df_clean['Billing_Bucket'] = pd.cut(
        df_clean['Days_Until_Due'],
        bins=[-100, -30, 0, 30, 60, 90, 180, 365],
        labels=["Very Early (<-30)", "Early (-30 to 0)", "On Time (0-30)", "Slight Delay (30-60)",
                "Late (60-90)", "Very Late (90-180)", "Extremely Late (180+)"],
        right=False
    )
    billing = df_clean.groupby('Billing_Bucket')['Days_To_Pay'].mean().reset_index()
    plot_line(billing, 'Billing_Bucket', 'Days_To_Pay', 'Billing Lead Time vs Days to Pay')

    # Heatmap of early payment by method and revenue type
    summary = df.groupby(['Payment method', 'Revenue Type'])['Early_Payment'].mean().reset_index()
    pivot = summary.pivot(index='Payment method', columns='Revenue Type', values='Early_Payment')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlGnBu", linewidths=0.5)
    plt.title("Early Payment Rate by Method & Revenue Type")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "early_payment_heatmap.png"))
    plt.close()

    return results


# Helper plotting functions
def plot_hist(df, column, title):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{column}_hist.png"))
    plt.close()


def plot_box(df, x, y, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{x}_{y}_boxplot.png"))
    plt.close()


def plot_line(df, x, y, title):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, marker='o')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{x}_{y}_lineplot.png"))
    plt.close()


def plot_bar(df, x, y, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{x}_{y}_barplot.png"))
    plt.close()


def plot_violin(df, x, y, title):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{x}_{y}_violinplot.png"))
    plt.close()


def plot_scatter(df, x, y, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{x}_{y}_scatter.png"))
    plt.close()