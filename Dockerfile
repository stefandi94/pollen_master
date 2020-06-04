FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-get install build-essential -y
RUN apt-get install checkinstall -y
RUN apt-get install libssl-dev -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.6 -y
RUN apt-get install python3.6-dev -y
RUN apt-get install python3-pip -y
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install wheel

WORKDIR /home/pollen

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

#CMD ["python3.6", "train.py"]
CMD ["python3.6", "app/manage.py", "runserver"]
