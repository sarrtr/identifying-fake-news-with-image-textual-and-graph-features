FROM python:3.12-slim 

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart pillow torchvision transformers torch lime grad-cam scikit-image streamlit requests opencv-python-headless

COPY ./code/deployment/api /app
COPY ./code/model/XAI_image.py /app
COPY ./code/model/XAI_text.py /app
COPY ./code/model/model_utils.py /app

COPY ./models /app/models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# uvicorn app:app --host 0.0.0.0 --port 8000

# cp /repo/project_deepfake/project/custom_model/code/deployment/api/app.py .

# cp /repo/project_deepfake/project/custom_model/code/deployment/app/streamlit_app.py .

# cp /repo/project_deepfake/project/custom_model/code/model/model_utils.py .

# cp /repo/project_deepfake/project/custom_model/code/model/XAI_image.py .
# cp /repo/project_deepfake/project/custom_model/code/model/XAI_text.py .
# cp -r /repo/project_deepfake/project/custom_model/models .