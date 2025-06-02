import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV or Excel file."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform preprocessing including cleaning, feature engineering, and date handling."""

    # Drop irrelevant columns
    df.drop(columns=[
        'Unnamed: 3', 'City.1', 'Country.1', 'Country code.1',
        'ZIP or postal code.1', 'State', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28'
    ], inplace=True, errors='ignore')

    # Convert date columns
    df['Invoice date'] = pd.to_datetime(df['Invoice date'], errors='coerce')
    df['Due date'] = pd.to_datetime(df['Due date'], errors='coerce')
    df['Receipt date'] = pd.to_datetime(df['Receipt date'], errors='coerce')

    # Drop rows with missing crucial dates
    df.dropna(subset=['Invoice date', 'Due date', 'Receipt date'], inplace=True)

    # Filter out 2022 data
    df = df[df['Invoice date'].dt.year != 2022]
    df.reset_index(drop=True, inplace=True)
    df = df[df['Invoice date'].dt.year != 2022].copy()

    # Create new features
    df.loc[:, 'Days_To_Pay'] = (df['Receipt date'] - df['Invoice date']).dt.days
    df.loc[:, 'Days_Until_Due'] = (df['Due date'] - df['Invoice date']).dt.days
    df.loc[:, 'Early_Payment'] = (df['Receipt date'] < df['Due date']).astype(int)

    # Handle missing values
    if 'City' in df.columns:
        df.loc[:, 'City'] = df['City'].fillna(df['City'].mode()[0])
    if 'State or territory' in df.columns:
        df.loc[:,'State or territory'] = df['State or territory'].fillna(df['State or territory'].mode()[0])
    if 'ZIP or postal code' in df.columns:
        df.loc[:,'ZIP or postal code'] = df['ZIP or postal code'].fillna('00000')
    if 'Item ID' in df.columns:
        df.loc[:,'Item ID'] = df['Item ID'].fillna('Unknown')

    # Drop high-cardinality ID-like columns (not useful for prediction)
    drop_cols = ['Customer ID', 'Item ID', 'Invoice ID']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Identify remaining categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # One-hot encode categorical features
    #df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
