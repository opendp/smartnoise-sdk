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

    schema = file("${path.root}/schema/pums.json")

    source_uris = [
        "gs://smartnoise-ci-bucket/PUMS.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.smartnoise_ci,
    ]
}

# PUMS_large dataset
resource "google_storage_bucket_object" "pums_large" {
  name   = "PUMS_large.csv"
  source = "../../../../datasets/PUMS_large.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_large" {
  dataset_id = google_bigquery_dataset.smartnoise_ci.dataset_id
  table_id   = "pums_large"
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

    schema = file("${path.root}/schema/pums.json")

    source_uris = [
        "gs://smartnoise-ci-bucket/PUMS_large.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.smartnoise_ci,
    ]
}

# PUMS_pid dataset 'datasets/PUMS_pid.csv'
resource "google_storage_bucket_object" "pums_pid" {
  name   = "PUMS_pid.csv"
  source = "../../../../datasets/PUMS_pid.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_pid" {
  dataset_id = google_bigquery_dataset.smartnoise_ci.dataset_id
  table_id   = "pums_pid"
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

    schema = file("${path.root}/schema/pums.json")

    source_uris = [
        "gs://smartnoise-ci-bucket/PUMS_pid.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.smartnoise_ci,
    ]
}

# PUMS_dup dataset 'datasets/PUMS_dup.csv'
resource "google_storage_bucket_object" "pums_dup" {
  name   = "PUMS_dup.csv"
  source = "../../../../datasets/PUMS_dup.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_dup" {
  dataset_id = google_bigquery_dataset.smartnoise_ci.dataset_id
  table_id   = "pums_dup"
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

    schema = file("${path.root}/schema/pums.json")

    source_uris = [
        "gs://smartnoise-ci-bucket/PUMS_dup.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.smartnoise_ci,
    ]
}
# PUMS_null dataset 'datasets/PUMS_null.csv'
resource "google_storage_bucket_object" "pums_null" {
  name   = "PUMS_null.csv"
  source = "../../../../datasets/PUMS_null.csv"
  bucket = google_storage_bucket.smartnoise_ci_bucket.name
}

resource "google_bigquery_table" "pums_null" {
  dataset_id = google_bigquery_dataset.smartnoise_ci.dataset_id
  table_id   = "pums_null"
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

    schema = file("${path.root}/schema/pums.json")

    source_uris = [
        "gs://smartnoise-ci-bucket/PUMS_null.csv"
    ]
  }

  depends_on = [
      google_bigquery_dataset.smartnoise_ci,
    ]
}
