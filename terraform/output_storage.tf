resource "aws_s3_bucket" "output_bucket" {
  bucket = "aaron-poke-output-bucket"

  tags = {
    Name        = "Pokedex Output Bucket"
    Environment = "Dev"
  }

  lifecycle {
    prevent_destroy = true
  }
}