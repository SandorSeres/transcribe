# Az alap kép, ami CUDA-t és Python-t is tartalmaz
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Frissítsük a Python-t a legújabb verzióra
RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.11 python3.11-distutils bash \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --config python3 \
    && apt-get clean

ENV PYTHONPATH=/app

# Telepítse a függőségeket a requirements.txt fájlból
COPY ./requirements.txt /app/
WORKDIR /app
RUN pip install --default-timeout=100 -r requirements.txt

# Másolja az alkalmazás kódját a konténerbe
COPY ./app.py /app/
COPY ./model_manager.py /app/
COPY ./static /app/static
COPY ./templates /app/templates
COPY ./.env /app/

# Az alkalmazás portjának nyitása
EXPOSE 8000

# Az alkalmazás indítása
CMD ["python3" , "/app/app.py"]