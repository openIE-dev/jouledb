terraform {
  required_version = ">= 1.5.0"
}

variable "cluster_name" {
  description = "Logical name of the JouleDB v0.9.0 deployment"
  type        = string
  default     = "jouledb-prod-v1"
}

variable "trust_tier" {
  description = "Progressive Trust & Compliance Tier: maker | consumer | soc2 | regulated | industrial | sovereign-mls"
  type        = string
  default     = "soc2"
  validation {
    condition     = contains(["maker", "consumer", "soc2", "regulated", "industrial", "sovereign-mls"], var.trust_tier)
    error_message = "trust_tier must be one of maker, consumer, soc2, regulated, industrial, or sovereign-mls."
  }
}

variable "regions" {
  description = "Multi-region Geo-Raft fault domains"
  type        = list(string)
  default     = ["us-east-1", "eu-central-1", "ap-northeast-1"]
}

locals {
  replica_count = var.trust_tier == "sovereign-mls" ? 5 : (var.trust_tier == "maker" || var.trust_tier == "consumer" ? 1 : 3)
  worm_enabled  = contains(["regulated", "industrial", "sovereign-mls"], var.trust_tier)
  pqc_mls       = var.trust_tier == "sovereign-mls"

  compliance_manifest = {
    version                 = "0.9.0"
    cluster_name            = var.cluster_name
    trust_tier              = var.trust_tier
    regions                 = var.regions
    raft_replicas           = local.replica_count
    sec_17a4_worm_storage   = local.worm_enabled
    fips_203_204_pqc_active = local.pqc_mls
    bell_lapadula_mls       = local.pqc_mls
    image                   = "ghcr.io/openie-dev/jouledb:0.9.0"
  }
}

resource "terraform_data" "jouledb_compliance_posture" {
  input = local.compliance_manifest
}

output "jouledb_cluster_posture" {
  description = "Rendered JouleDB v0.9.0 Geo-Raft & Compliance posture"
  value       = terraform_data.jouledb_compliance_posture.output
}
