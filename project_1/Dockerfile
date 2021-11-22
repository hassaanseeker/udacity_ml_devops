FROM python:3.8.8

WORKDIR /ChurnPredictor
COPY requirements.txt .

RUN pip install -r requirements.txt


COPY ./project ./project

WORKDIR /ChurnPredictor/project

CMD ["python3", "-u", "./main.py"]
