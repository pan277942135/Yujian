FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /worker
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip git git-lfs libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/open-mmlab/PowerPaint.git /opt/PowerPaint
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision && python3 -m pip install -r requirements.txt
COPY . .
ENV PYTHONPATH=/opt/PowerPaint:/worker PORT=8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port 8080"]
