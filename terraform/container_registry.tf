resource "aws_ecr_repository" "poke_ecr" {
  name                 = "poke-ecr"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle {
    prevent_destroy = true
  }
}