# Kubernetes Deployment Script for Heart Disease Predictor

# Prerequisites:
# 1. Docker Desktop with Kubernetes enabled, OR
# 2. Minikube installed and running

# For Docker Desktop:
# - Open Docker Desktop
# - Go to Settings > Kubernetes
# - Enable "Enable Kubernetes"
# - Wait for cluster to start

# For Minikube:
# minikube start
# minikube dashboard (optional)

# Verify cluster is running:
kubectl cluster-info

# Build and load Docker image into cluster:
# For Minikube:
# minikube docker-env
# eval $(minikube -p minikube docker-env)
# docker build -t heart-disease-predictor:v1.0 .

# For Docker Desktop, the image is already available

# Deploy the application:
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml

# Check deployment status:
kubectl get pods
kubectl get services

# Get service URL:
# For Minikube:
# minikube service heart-disease-service --url

# For Docker Desktop:
# kubectl get service heart-disease-service

# Test the deployed API:
# curl http://<service-url>/health
# curl -X POST http://<service-url>/predict -H "Content-Type: application/json" -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'

# View logs:
kubectl logs -l app=heart-disease-predictor

# Scale the deployment:
kubectl scale deployment heart-disease-predictor --replicas=3

# Clean up:
kubectl delete -f k8s-deployment.yaml
kubectl delete -f k8s-service.yaml