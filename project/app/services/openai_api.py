import openai
import os

# Set your OpenAI API key (you can also load this from environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_openai(question: str, context: str) -> str:
    """
    Ask a question to OpenAI using the provided context (EDA, predictions, etc.).
    
    Parameters:
    - question: The user’s question.
    - context: A string summarizing EDA results, predictions, model performance, etc.

    Returns:
    - OpenAI's response as a string.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data science assistant. You help interpret EDA results, machine learning model outputs, and predictive insights."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ],
            temperature=0,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error: {str(e)}"