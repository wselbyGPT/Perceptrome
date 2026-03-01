#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA_FILE="${USER_DATA_FILE:-${SCRIPT_DIR}/ec2_bootstrap.sh}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"

: "${AMI_ID:?AMI_ID is required}"
: "${KEY_NAME:?KEY_NAME is required}"
: "${SECURITY_GROUP_ID:?SECURITY_GROUP_ID is required}"
: "${SUBNET_ID:?SUBNET_ID is required}"
: "${IAM_INSTANCE_PROFILE:?IAM_INSTANCE_PROFILE is required}"

if [[ ! -f "${USER_DATA_FILE}" ]]; then
  echo "User-data file not found: ${USER_DATA_FILE}" >&2
  exit 1
fi

run_instances_args=(
  --region "${AWS_REGION}"
  --image-id "${AMI_ID}"
  --instance-type "${INSTANCE_TYPE}"
  --key-name "${KEY_NAME}"
  --security-group-ids "${SECURITY_GROUP_ID}"
  --subnet-id "${SUBNET_ID}"
  --iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}"
  --user-data "file://${USER_DATA_FILE}"
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=perceptrome}]'
  --count 1
  --query 'Instances[0].InstanceId'
  --output text
)

INSTANCE_ID="$(aws ec2 run-instances "${run_instances_args[@]}")"
echo "INSTANCE_ID=${INSTANCE_ID}"

echo "Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"

echo "Resolving public IP..."
EC2_PUBLIC_IP="$(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --region "${AWS_REGION}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

echo "EC2_PUBLIC_IP=${EC2_PUBLIC_IP}"

if [[ -n "${HOSTED_ZONE_ID:-}" ]]; then
  RECORD_NAME="${RECORD_NAME:-perceptrome.com}"
  TTL="${TTL:-300}"
  echo "Route53 variables detected; upserting DNS record ${RECORD_NAME}"
  HOSTED_ZONE_ID="${HOSTED_ZONE_ID}" \
  RECORD_NAME="${RECORD_NAME}" \
  EC2_PUBLIC_IP="${EC2_PUBLIC_IP}" \
  TTL="${TTL}" \
  AWS_REGION="${AWS_REGION}" \
  "${SCRIPT_DIR}/create_route53_record.sh"
else
  echo "HOSTED_ZONE_ID not set; skipping Route53 record update"
fi
