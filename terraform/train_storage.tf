resource "aws_s3_bucket" "poke_bucket" {
  bucket = "aaron-poke-training-bucket"

  tags = {
    Name        = "Pokedex Training Bucket"
    Environment = "Dev"
  }

  lifecycle {
    prevent_destroy = true
  }
}