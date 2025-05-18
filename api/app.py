# Basic FastAPI app
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {"message": "NLP Classification API is running"}