terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "model_factory" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age            = 30
      matches_prefix = ["temp/"]
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_service_account" "model_factory" {
  account_id   = "yujian-model-factory"
  display_name = "YuJian Model Factory"
}

resource "google_storage_bucket_iam_member" "writer" {
  bucket = google_storage_bucket.model_factory.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.model_factory.email}"
}

output "bucket_uri" {
  value = "gs://${google_storage_bucket.model_factory.name}"
}

output "service_account" {
  value = google_service_account.model_factory.email
}
