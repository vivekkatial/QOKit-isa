# INFORMS Publication


### Running with Docker

```
docker build -t qokit .
```

Run the container with the following command:
```
docker run -p 8888:8888 qokit
```

Run the container with the following command to mount the current directory:
```
docker run -v $(pwd):/app -p 8888:8888 qokit
```

Run the instance and add .env file and mount the current directory:
```
docker run -v $(pwd):/app --env-file .env -p 8888:8888 qokit
```

#### MaxCut

For MaxCut, the datasets in `qokit/assets/maxcut_datasets` must be inflated


## Big thanks to the creators of the following libraries:
- [Qiskit](https://qiskit.org/)
- [QOKit](https://github.com/jpmorganchase/QOKit)