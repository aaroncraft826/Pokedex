variable "REGION" {
  default = "us-east-1"
}

variable "ZONE1" {
  default = "us-east-1a"
}

variable "ZONE2" {
  default = "us-east-1b"
}

variable "instance_ami" {
  default = "ami-04b4f1a9cf54c11d0"
}

variable "instance_type" {
  default = "t2.micro"
}

variable "cluster_name" {
  default = "poke-cluster"
}