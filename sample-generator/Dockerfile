FROM python:3.11-slim

WORKDIR /app

COPY sample-generator.py /app

RUN pip install flask

EXPOSE 8000

CMD ["python", "sample-generator.py"]
