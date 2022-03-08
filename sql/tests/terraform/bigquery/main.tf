terraform {
  required_version = ">= 0.13.1"
  backend "local" {}
  required_providers {
  google = {
      source = "hashicorp/google"
    }
  }
}

provider "google" {
 project = var.project_id
 region  = var.region
}

resource "google_storage_bucket" "smartnoise_ci_bucket" {
  name          = var.bucket_name
  force_destroy = true
  location      = var.region
}

resource "google_bigquery_dataset" "smartnoise_ci" {
  project    = var.project_id
  dataset_id = "smartnoise_ci"
  labels = {
    "environment" = "smartnoise-sdk-ci"
  }
  delete_contents_on_destroy = true
}

# PUMS dataset
resource "google_storage_bucket_object" "pums" {
  name   = "PUMS.csv"
  source = "../../../../datasets/PUMS.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums" {
  dataset_id = google_bigquery_dataset.smartnoise_ci.dataset_id
  table_id   = "pums"
  deletion_protection = false

  labels = {
    "environment" = "smartnoise-sdk-ci"
  }

  external_data_configuration {
    autodetect    = true
    source_format = "CSV"

    csv_options {
      quote = ""
      skip_leading_rows = 1
    }

    schema = <<EOF
[
  {
    "name": "age",
    "type": "INTEGER",
    "mode": "NULLABLE"
  },
  {
    "name": "sex",
    "type": "INTEGER",
    "mode": "NULLABLE"
  },
  {
    "name": "educ",
    "type": "INTEGER",
    "mode": "NULLABLE"
  },
  {
    "name": "race",
    "type": "INTEGER",
    "mode": "NULLABLE"
  },
  {
    "name": "income",
    "type": "FLOAT",
    "mode": "NULLABLE"
  },
  {
    "name": "married",
    "type": "INTEGER",
    "mode": "NULLABLE"
  }
]
EOF

    source_uris = [
        "gs://smartnoise-ci-bucket/PUMS.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.smartnoise_ci,
    ]
}
