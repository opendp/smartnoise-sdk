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

resource "google_bigquery_dataset" "pums" {
  project    = var.project_id
  dataset_id = "PUMS"
  location   = var.region
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
  dataset_id = google_bigquery_dataset.pums.dataset_id
  table_id   = "PUMS"
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

    schema = file("${path.root}/schema/PUMS.json")

    source_uris = [
        "gs://${var.bucket_name}/PUMS.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.pums,
    ]
}

# PUMS_large dataset
resource "google_storage_bucket_object" "pums_large" {
  name   = "PUMS_large.csv"
  source = "../../../../datasets/PUMS_large.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_large" {
  dataset_id = google_bigquery_dataset.pums.dataset_id
  table_id   = "PUMS_LARGE"
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

    schema = file("${path.root}/schema/PUMS_large.json")

    source_uris = [
        "gs://${var.bucket_name}/PUMS_large.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.pums,
    ]
}

# PUMS_pid dataset 'datasets/PUMS_pid.csv'
resource "google_storage_bucket_object" "pums_pid" {
  name   = "PUMS_pid.csv"
  source = "../../../../datasets/PUMS_pid.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_pid" {
  dataset_id = google_bigquery_dataset.pums.dataset_id
  table_id   = "PUMS_PID"
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

    schema = file("${path.root}/schema/PUMS_pid.json")

    source_uris = [
        "gs://${var.bucket_name}/PUMS_pid.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.pums,
    ]
}

# PUMS_dup dataset 'datasets/PUMS_dup.csv'
resource "google_storage_bucket_object" "pums_dup" {
  name   = "PUMS_dup.csv"
  source = "../../../../datasets/PUMS_dup.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_dup" {
  dataset_id = google_bigquery_dataset.pums.dataset_id
  table_id   = "PUMS_DUP"
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

    schema = file("${path.root}/schema/PUMS_dup.json")

    source_uris = [
        "gs://${var.bucket_name}/PUMS_dup.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.pums,
    ]
}
# PUMS_null dataset 'datasets/PUMS_null.csv'
resource "google_storage_bucket_object" "pums_null" {
  name   = "PUMS_null.csv"
  source = "../../../../datasets/PUMS_null.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_null" {
  dataset_id = google_bigquery_dataset.pums.dataset_id
  table_id   = "PUMS_NULL"
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

    schema = file("${path.root}/schema/PUMS_null.json")

    source_uris = [
        "gs://${var.bucket_name}/PUMS_null.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.pums,
    ]
}
