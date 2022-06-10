variable "project_id" {
  description = "The target project"
  type        = string
}

variable "region" {
  description = "The region where resources are created => europe-west2"
  type        = string
  default     = "us-west1"
}

variable "bucket_name" {
  description = "The zone in the europe-west region for resources"
  type        = string
}
