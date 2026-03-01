# AWS deployment bundle

This folder contains scripts for provisioning and deploying Perceptrome on a single Ubuntu EC2 instance behind Nginx.

## Prerequisites

- AWS CLI v2 installed and authenticated (`aws configure` or equivalent profile/env).
- IAM permissions for:
  - `ec2:RunInstances`, `ec2:DescribeInstances`, `ec2:TerminateInstances`, `ec2:CreateTags`, `ec2:Describe*`
  - `iam:PassRole` (for your instance profile)
  - `route53:ChangeResourceRecordSets` and `route53:ListHostedZones` (if using Route53)
- An existing hosted zone in Route53 for `perceptrome.com`.
- EC2 networking setup:
  - VPC/subnet for the instance
  - Security group allowing inbound TCP `22`, `80`, and `443`
- A key pair (`KEY_NAME`) available in the target region.

## Files

- `ec2_bootstrap.sh` — user-data/first-run bootstrap for app + systemd + Nginx (+ optional certbot).
- `provision_ec2.sh` — launches EC2 with bootstrap as user-data, waits for public IP, optionally updates Route53.
- `create_route53_record.sh` — UPSERTs Route53 A record to the instance public IP.

## Copy/paste run sequence

From repository root:

```bash
chmod +x infra/aws/*.sh
cd infra/aws
cp .env.example .env
$EDITOR .env
set -a; source .env; set +a
cd ../..
```

Launch an EC2 instance:

```bash
./infra/aws/provision_ec2.sh
```

Wait for the instance and application readiness (typically 2-5 minutes after `instance-running`):

```bash
# Replace with output INSTANCE_ID if needed
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"
```

Create or update Route53 manually (if you skipped it during provisioning):

```bash
export EC2_PUBLIC_IP="$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$AWS_REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

./infra/aws/create_route53_record.sh
```

Verify HTTP response:

```bash
curl -I http://perceptrome.com
```


## Environment file and secrets

- Commit `infra/aws/.env.example`: it should contain only placeholders/defaults and non-sensitive identifiers.
- Do **not** commit `infra/aws/.env`: this can contain account- or environment-specific values.
- Treat these as sensitive and keep them out of git:
  - `NCBI_API_KEY`
  - `LETSENCRYPT_EMAIL`
  - any private/internal values you add (for example private repo URLs or credentials)
- Usually safe to commit as examples (but still environment-specific):
  - `AWS_REGION`, `AMI_ID`, `INSTANCE_TYPE`
  - `KEY_NAME`, `SECURITY_GROUP_ID`, `SUBNET_ID`, `HOSTED_ZONE_ID`, `RECORD_NAME`
  - toggles/defaults like `ENABLE_TLS`, `APP_DIR`, `EC2_BOOTSTRAP_USER`

## Rollback

Terminate instance:

```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"
```

Remove DNS record (change action from UPSERT to DELETE):

```bash
cat > /tmp/delete-record.json <<JSON
{
  "Comment": "Delete perceptrome.com A record",
  "Changes": [
    {
      "Action": "DELETE",
      "ResourceRecordSet": {
        "Name": "${RECORD_NAME:-perceptrome.com}",
        "Type": "A",
        "TTL": ${TTL:-300},
        "ResourceRecords": [{"Value": "${EC2_PUBLIC_IP}"}]
      }
    }
  ]
}
JSON

aws route53 change-resource-record-sets \
  --hosted-zone-id "$HOSTED_ZONE_ID" \
  --change-batch file:///tmp/delete-record.json \
  --region "$AWS_REGION"
```
