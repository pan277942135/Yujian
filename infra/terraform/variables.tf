variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "GCS bucket region"
  type        = string
  default     = "asia-east1"
}

variable "bucket_name" {
  description = "Globally unique GCS bucket name"
  type        = string
}
