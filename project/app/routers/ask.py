from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


router = APIRouter()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_insights(request: QuestionRequest):
    try:
        # Load EDA results
        eda_path = "data/eda_results.pkl"
        if not os.path.exists(eda_path):
            raise HTTPException(status_code=400, detail="EDA results not found. Run /analyze first.")
        eda_results = pd.read_pickle(eda_path)

        # Load dataset
        df_path = "data/active_df.pkl"
        if not os.path.exists(df_path):
            raise HTTPException(status_code=400, detail="Dataset not found. Upload data first.")
        df = pd.read_pickle(df_path)

        # Shorten dataset for context
        df_sample = df.head(5).to_string(index=False)

        # Load visualization summaries
        viz_folder = 'outputs'
        viz_descriptions = []
        if os.path.exists(viz_folder):
            for file in os.listdir(viz_folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    description = file.replace("_", " ").replace(".png", "").replace(".jpg", "").title()
                    viz_descriptions.append(f"- {file}: {description}")
        else:
            viz_descriptions.append("⚠️ Visualizations folder not found.")

        viz_summary = "\n".join(viz_descriptions) if viz_descriptions else "No visualizations found."

        # Compose context string
        context = f"""
You are analyzing a dataset of invoices and payment behavior. Here's the context:

📊 Sample Data:
{df_sample}

📈 EDA Insights:
- Summary Statistics: {eda_results.get("summary_stats")}
- Early Payment Ratio: {eda_results.get("early_payment_ratio")}
- Payment Terms Count: {eda_results.get("payment_terms_count")}
- Payment Terms Stats: {eda_results.get("payment_terms_stats")}
- ANOVA P-Value: {eda_results.get("anova_p")}
- Kruskal-Wallis P-Value: {eda_results.get("kruskal_p")}
- Correlation Between Days Until Due & Days To Pay: {eda_results.get("due_pay_correlation")}

📂 Visualizations:
Plots are saved in the outputs folder showing histograms, distributions, and relationships:
{viz_summary}
"""

        # Send to OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analyst interpreting EDA and business dataset insights."},
                {"role": "user", "content": context},
                {"role": "user", "content": request.question}
            ],
            temperature=0,
        )

        answer = completion.choices[0].message.content
        return {"question": request.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
