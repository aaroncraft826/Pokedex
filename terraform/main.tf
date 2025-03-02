terraform {
    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = "~> 5.89"
        }
    }

    required_version = ">= 1.2.0"
}

provider "aws" {
    region = "us-east-1"
}

# resource "aws_instance" "training_server" {
#     ami                         = var.instance_ami
#     instance_type               = var.instance_type
#     availability_zone           = var.ZONE1
#     security_groups             = [aws_security_group.poke_sg.id]
#     associate_public_ip_address = true
#     subnet_id                   = aws_subnet.public_subnet_1.id
#     count = 0

#     ### Install Docker
#     user_data = <<-EOF
#     #!/bin/bash
#     curl -fsSL https://get.docker.com -o get-docker.sh
#     sudo sh get-docker.sh
#     sudo groupadd docker
#     sudo usermod -aG docker ubuntu
#     newgrp docker
#     sudo timedatectl set-timezone America/New_York
#     EOF

#     tags = {
#         Name = "pokedex-training-server-${count.index}"
#     }

#     depends_on = [ aws_s3_bucket.output_bucket, aws_s3_bucket.train_bucket, aws_ecr_repository.poke_ecr ]
# }