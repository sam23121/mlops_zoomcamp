FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY ./starter.py /app

RUN pip install -U pip
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

CMD [ "python", "starter.py", "2023", "5" ] 



