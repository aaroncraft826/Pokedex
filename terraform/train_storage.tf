import {
  to = aws_s3_bucket.train_bucket
  id = "aaron-poke-training-bucket"
}

resource "aws_s3_bucket" "train_bucket" {
  bucket = "aaron-poke-training-bucket"

  tags = {
    Name        = "Pokedex Training Bucket"
    Environment = "Dev"
  }

  lifecycle {
    prevent_destroy = true
  }
}