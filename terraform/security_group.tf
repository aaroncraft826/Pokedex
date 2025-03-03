resource "aws_security_group" "poke_sg" {
  name   = "SSH + Port 3005 for API"
  vpc_id = module.vpc.vpc_id
 
  ingress {
    cidr_blocks = [
      "0.0.0.0/0"
    ]
    from_port   = 0
    to_port     = 65535
    protocol  = "tcp"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}