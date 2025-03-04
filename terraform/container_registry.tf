import {
  to = aws_ecr_repository.poke_ecr
  id = "aaron-poke-ecr"
}

resource "aws_ecr_repository" "poke_ecr" {
  name                 = "aaron-poke-ecr"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle {
    prevent_destroy = true
  }
}