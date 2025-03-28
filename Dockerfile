FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /workspace

COPY requirements.txt .

RUN apt-get update

RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

CMD ["ls"]