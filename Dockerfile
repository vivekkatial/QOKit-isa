FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV JUPYTER_ENABLE_LAB=yes

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ python3-dev git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && \
    pip install .

RUN pip install notebook

# Install things in requirements-isa.txt
RUN pip install -r requirements-isa.txt

EXPOSE 8888

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
# just run the so i can ssh into the container
CMD ["tail", "-f", "/dev/null"]
