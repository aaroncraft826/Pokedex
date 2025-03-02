module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"

  cluster_name    = "poke-cluster"
  cluster_version = "1.31"

  iam_role_arn = aws_iam_role.cluster.arn

  # Optional
  cluster_endpoint_public_access = true

  # Optional: Adds the current caller identity as an administrator via cluster access entry
  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    poke_node_group = {
        instance_types = ["g5.xlarge"]
        min_size = 2
        max_size = 2
        desired_size = 2
    }
  }

  vpc_id     = aws_vpc.poke_vpc.id
  subnet_ids = [aws_subnet.public_subnet_1.id, aws_subnet.private_subnet_1.id, aws_subnet.public_subnet_2.id, aws_subnet.private_subnet_2.id]

  tags = {
    Environment = "dev"
    Terraform   = "true"
  }
}

resource "aws_iam_role" "cluster" {
  name = "poke-cluster"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster.name
}