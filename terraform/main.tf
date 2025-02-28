terraform {
    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = "~> 4.16"
        }
    }

    required_version = ">= 1.2.0"
}

provider "aws" {
    region = "us-east-1"
}

resource "aws_instance" "training_server" {
    ami = "ami-830c94e3"
    instance_type = "t2.micro"
    count = 3

    tags = {
        Name = "pokedex-training-server-${count.index}"
    }

    depends_on = [ aws_s3_bucket.output_bucket, aws_s3_bucket.train_bucket, aws_ecr_repository.poke_ecr ]
}