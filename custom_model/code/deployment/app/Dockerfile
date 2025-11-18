FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests numpy

COPY ./code/deployment/app /app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]