# Build

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker build . -t velmira/kcd-sofia-sample-generator:v0.1
```

# Push to Docker Hub

```bash
docker push velmira/kcd-sofia-sample-generator:v0.1
```

# Apply in K8s cluster

```bash
kubectl apply -f ./deployment.yaml
```

# Restart all generator pods

```bash
kubectl delete pods -n sample-generator -l 'app=sample-metrics'
```

# Access to Prometheus

```bash
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
```

This command never exits unless the computer sleeps.

Then you can open web browser on [localhost:9090](http://localhost:9090/), go to the Graph tab and issue the query manually `rate(container_cpu_usage_seconds_total{pod!="",namespace="default",container!="", kubernetes_io_hostname!=""}[5m])`.