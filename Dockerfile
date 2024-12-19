FROM python:3.11-slim

WORKDIR /tum-adlr-03

COPY . /tum-adlr-03

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "multi_class"]
