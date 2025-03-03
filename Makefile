docker-login:
	aws ecr get-login-password | docker login -u AWS --password-stdin "https://$(aws sts get-caller-identity --query 'Account' --output text).dkr.ecr.$(aws configure get region).amazonaws.com"

docker-build:
	docker build -f Dockerfile --no-cache -t aaron-poke-ecr .
	
docker-tag:
	docker tag aaron-poke-ecr:latest <account-name>.dkr.ecr.us-east-1.amazonaws.com/aaron-poke-ecr:latest

docker-push:
	docker push <account-name>.dkr.ecr.us-east-1.amazonaws.com/aaron-poke-ecr:latest

deploy: 
	kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.10/nvidia-device-plugin.yml
	kubectl apply -f deploy/daemonset.yaml
