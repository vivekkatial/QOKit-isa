FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV JUPYTER_ENABLE_LAB=yes

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    g++ \
    build-essential \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    pkg-config \
    libsuitesparse-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Check if umfpack.h is present
RUN find /usr -name "umfpack.h"

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib
ENV C_INCLUDE_PATH=/usr/include:/usr/local/include
ENV CPPFLAGS="-I/usr/include/suitesparse"

COPY . /app

RUN pip install --upgrade pip && \
    pip install .

RUN pip install notebook

# Install things in requirements-isa.txt
RUN pip install -r requirements-isa.txt

EXPOSE 8888

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
# Can run this and then ssh into container using (+ mounting the volume)
# docker run -it -p 8888:8888 -v $(pwd):/app <image_id> /bin/bash
# docker exec -it <container_id> /bin/bash
CMD ["tail", "-f", "/dev/null"]
