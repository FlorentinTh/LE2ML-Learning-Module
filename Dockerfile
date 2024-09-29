FROM python:3.13.0rc2-slim

COPY . .

RUN mv prod.requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD [ "python3", "-u", "./src/main.py" ]
