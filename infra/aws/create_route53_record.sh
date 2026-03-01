#!/usr/bin/env bash
set -euo pipefail

: "${HOSTED_ZONE_ID:?HOSTED_ZONE_ID is required}"
: "${EC2_PUBLIC_IP:?EC2_PUBLIC_IP is required}"

RECORD_NAME="${RECORD_NAME:-perceptrome.com}"
TTL="${TTL:-300}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

cat > "${TMP_FILE}" <<JSON
{
  "Comment": "UPSERT ${RECORD_NAME} -> ${EC2_PUBLIC_IP}",
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "${RECORD_NAME}",
        "Type": "A",
        "TTL": ${TTL},
        "ResourceRecords": [
          {
            "Value": "${EC2_PUBLIC_IP}"
          }
        ]
      }
    }
  ]
}
JSON

aws route53 change-resource-record-sets \
  --hosted-zone-id "${HOSTED_ZONE_ID}" \
  --change-batch "file://${TMP_FILE}" \
  --region "${AWS_REGION}"

echo "Route53 A record upsert submitted for ${RECORD_NAME} -> ${EC2_PUBLIC_IP}"
