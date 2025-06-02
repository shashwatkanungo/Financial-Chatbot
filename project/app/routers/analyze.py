from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from app.services.eda import analyze_data
from app.services.data_preprocessor import preprocess_data
from openai import OpenAI
import base64
import json
from glob import glob
from dotenv import load_dotenv

load_dotenv()


router = APIRouter()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def encode_images_to_base64(image_folder: str):
    image_files = sorted(glob(os.path.join(image_folder, "*.png")))
    base64_images = []
    for img_path in image_files:
        with open(img_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode("utf-8")
            base64_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_str}"
                }
            })
    return base64_images

@router.post("/analyze")
def analyze_endpoint():
    try:
        df_path = "data/active_df.pkl"
        if not os.path.exists(df_path):
            raise HTTPException(status_code=400, detail="No dataset uploaded")

        df = pd.read_pickle(df_path)
        df = preprocess_data(df)  # Preprocess the data

        results = analyze_data(df)

        # Optionally save results for ask endpoint
        pd.to_pickle(results, "data/eda_results.pkl")

        # Encode EDA images
        images = encode_images_to_base64("outputs/eda")

        # Prepare JSON summary for context
        json_context = json.dumps(results, indent=2)


        # prepare the openAI summary for context
        response = client.chat.completions.create(
            model= "gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior data analyst helping interpret exploratory data analysis (EDA) for a financial dataset. "
                        "You are given a JSON summary of key metrics and also a set of EDA visualizations. "
                        "You must return a detailed, well-structured report with separate sections:\n"
                        "1. Summary Statistics\n"
                        "2. Behavioral Patterns\n"
                        "3. Visualization Insights\n"
                        "4. Business Implications\n"
                        "Make sure the tone is analytical and recommendations are data-driven."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here are the structured EDA results (in JSON format):"},
                        {"type": "text", "text": json_context},
                        {"type": "text", "text": "Here are the related EDA charts. Please describe what each visual represents and summarize any visible patterns:"},
                        *images
                    ]
                }
            ]
        )

        summary = response.choices[0].message.content
        return {"status":"success", "insights": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
