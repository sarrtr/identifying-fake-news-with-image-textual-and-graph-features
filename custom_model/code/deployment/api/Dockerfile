FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi uvicorn[standard] python-multipart pillow torchvision transformers torch lime grad-cam scikit-image streamlit requests

COPY ./code/deployment/api /app
COPY ./models /app
COPY ./code/model/XAI_image.py /app
COPY ./code/model/XAI_text.py /app
COPY ./code/model/model_utils.py /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]