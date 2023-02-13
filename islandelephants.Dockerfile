FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONUNBUFFERED=1

WORKDIR /
COPY islandelephants /islandelephants
COPY requirements.txt /requirements.txt
COPY islandelephants.py /islandelephants.py

RUN mkdir /scratch && \
    chmod og+rwx /

RUN apt-get -y install libsndfile1

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --requirement /requirements.txt

ENTRYPOINT ["python", "islandelephants.py"]
