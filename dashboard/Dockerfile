# Use official Python image
FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords punkt wordnet omw-1.4

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
