from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import os

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()

    try:
        if file_ext == '.csv':
            df = pd.read_csv(file.file)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Save to temp location (for analysis and ask)
        df.to_pickle("data/active_df.pkl")

        return {"message": "File uploaded successfully", "rows": len(df), "columns": list(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
