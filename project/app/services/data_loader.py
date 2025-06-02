import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV or Excel file.
    Supported formats: .csv, .xlsx, .xls
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")
    
    return df

def save_uploaded_file(file, upload_folder="uploads/"):
    """
    Save uploaded file to a specified folder.
    The folder is created if it doesn't exist.
    """
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    return file_path
