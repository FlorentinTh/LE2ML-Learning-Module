FROM python:3.9-slim

COPY . .

RUN mv prod.requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD [ "python3", "-u", "./src/main.py" ]
