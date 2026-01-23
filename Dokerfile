FROM python:3.10

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860

CMD ["python", "run.py"]
