from fastapi import FastAPI
from pydantic import BaseModel      
import uvicorn  
import joblib
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from fastapi import HTTPException
from pyngrok import ngrok



logger = logging.getLogger("uvicorn.error")

import os

# Get the folder where this script is located
BASE_DIR = os.path.dirname(__file__)



# Combine with your model filename
MODEL_PATH = os.path.join(BASE_DIR, "Fake_news.sav")

VECTOR_PATH = os.path.join(BASE_DIR, "vector.sav")

# Load the model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTOR_PATH, "rb") as f:
    vector = pickle.load(f)

app = FastAPI()

class Input(BaseModel):
    combine: str


# Load the model and vectorizer
model= pickle.load(open("C:/Users/HP/Projects For Job/Fake_news.sav", 'rb'))
vector = pickle.load(open("C:/Users/HP/Projects For Job/vector.sav", "rb"))


#Creating the API endpoint
@app.post("/predict")   

def predict(input: Input):
    try:
        input_data = input.combine
        input_data = [input_data]

        
        input_data = vector.transform(input_data)
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start ngrok tunnel for public URL
    public_url = ngrok.connect(8000)
    print(f"Public URL: {public_url}")

    uvicorn.run(app, port=8000)
