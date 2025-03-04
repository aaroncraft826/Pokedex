docker-login:
	aws ecr get-login-password | docker login -u AWS --password-stdin "https://$(aws sts get-caller-identity --query 'Account' --output text).dkr.ecr.$(aws configure get region).amazonaws.com"

docker-build:
	docker build -f Dockerfile --no-cache -t aaron-poke-ecr .
	
docker-tag:
	docker tag aaron-poke-ecr:latest <acc>.dkr.ecr.us-east-1.amazonaws.com/aaron-poke-ecr:latest

docker-push:
	docker push <acc>.dkr.ecr.us-east-1.amazonaws.com/aaron-poke-ecr:latest

make docker:
	docker build -f Dockerfile --no-cache -t aaron-poke-ecr .
	docker tag aaron-poke-ecr:latest <acc>.dkr.ecr.us-east-1.amazonaws.com/aaron-poke-ecr:latest
	docker push <acc>.dkr.ecr.us-east-1.amazonaws.com/aaron-poke-ecr:latest

kube-login:
	aws eks update-kubeconfig --region us-east-1 --name poke-cluster

deploy: 
	kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
	kubectl apply -f deployment/daemonset.yaml

destroy: 
	kubectl delete -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
	kubectl delete -f deployment/daemonset.yaml
