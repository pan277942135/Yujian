FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG POWERPAINT_COMMIT=5b4c3d52291709fcec2a1870d987da693fd3549c
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 POWERPAINT_COMMIT=${POWERPAINT_COMMIT}
WORKDIR /worker
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip git git-lfs libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/open-mmlab/PowerPaint.git /opt/PowerPaint \
    && cd /opt/PowerPaint \
    && git checkout "${POWERPAINT_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${POWERPAINT_COMMIT}"
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 \
    && python3 -m pip install -r requirements.txt
COPY . .
ENV PYTHONPATH=/opt/PowerPaint:/worker PORT=8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port 8080"]
