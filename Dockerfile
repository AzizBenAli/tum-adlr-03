FROM python:3.11-slim

WORKDIR /tum-adlr-03

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

COPY . /tum-adlr-03

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "multi_class"]
