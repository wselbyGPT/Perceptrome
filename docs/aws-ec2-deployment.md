# Deploying Perceptrome Web on AWS EC2

This guide walks through a single‑node production deployment of `perceptrome_web`
on Amazon EC2, using:

- **EC2** for the application host (Docker + docker‑compose).
- **RDS for PostgreSQL** as the user/auth/run database.
- **ALB (optional)** to terminate TLS, or **nginx + Let's Encrypt** on the
  instance itself.
- **SES (optional)** for password‑reset / invitation email.
- **S3 (optional)** for run artifacts / model checkpoints.

The deployment is built around the repo's `docker-compose.prod.yml`, which
brings up three services: `migrate` (one‑shot Alembic), `api` (FastAPI +
baked‑in Vite SPA), and `nginx` (reverse proxy).

---

## 1. Architecture at a glance

```
                   ┌────────────────────────────┐
   internet ─►  ┌──┤  ALB (TLS) — optional      │
                │  └────────────────────────────┘
                │            │ HTTP :80
                ▼            ▼
          ┌─────────────────────────────────┐
          │  EC2 instance                   │
          │                                 │
          │   nginx :80/:443  (container)   │
          │       │                         │
          │       ▼                         │
          │   api :8000      (container)    │
          │       │  (Alembic migrate)      │
          │       ▼                         │
          └──────────────────────────────────┘
                  │ TCP :5432
                  ▼
          ┌──────────────────────┐
          │  RDS PostgreSQL 16   │
          └──────────────────────┘
```

The API container ships the built React/Vite SPA inside it (see
`perceptrome_web/Dockerfile`), so nginx only needs to reverse‑proxy to
`api:8000` — no separate static bucket required.

---

## 2. Prerequisites

On your workstation:

- AWS account with admin (or sufficiently scoped) IAM credentials.
- AWS CLI v2 configured (`aws configure`).
- An SSH key pair you control (or one created in EC2 console).
- A domain name (Route 53 or external DNS) if you want HTTPS.

On the EC2 instance you will install:

- Docker Engine + the Docker Compose plugin
- `git` and `make` (optional but convenient)

The repo's only required runtime dependency is Docker; Python is **not**
needed on the host because everything runs inside the `perceptrome_web`
image.

---

## 3. Provision AWS resources

### 3.1 VPC and security groups

If you do not already have one, the default VPC is fine for a single‑node
deployment. Create three security groups:

| Name | Inbound | Source |
|------|---------|--------|
| `perceptrome-alb-sg` (only if using ALB) | 80, 443 | `0.0.0.0/0` |
| `perceptrome-ec2-sg` | 22 from your IP; 80 from ALB SG (or `0.0.0.0/0` if no ALB) | — |
| `perceptrome-rds-sg` | 5432 | `perceptrome-ec2-sg` |

Avoid putting `0.0.0.0/0` on port 22 — restrict SSH to your office/VPN range
or use **EC2 Instance Connect** / **Session Manager** instead.

### 3.2 RDS PostgreSQL

Create an RDS instance:

```bash
aws rds create-db-instance \
  --db-instance-identifier perceptrome-db \
  --db-instance-class db.t4g.small \
  --engine postgres --engine-version 16.4 \
  --allocated-storage 20 --storage-type gp3 \
  --master-username perceptrome \
  --master-user-password 'CHANGE_ME_STRONG_PASSWORD' \
  --db-name perceptrome \
  --vpc-security-group-ids sg-xxxxxxxx \
  --backup-retention-period 7 \
  --no-publicly-accessible
```

Notes:

- Use a **private** RDS instance (`--no-publicly-accessible`); the EC2
  instance reaches it inside the VPC.
- `db.t4g.small` is the smallest sensible size; scale up if you expect
  heavy run history or BioAST graph storage.
- Note the endpoint: `perceptrome-db.xxxx.us-east-1.rds.amazonaws.com`.

### 3.3 EC2 instance

A `t3.large` (2 vCPU / 8 GiB) is the practical minimum if you want to
actually run small training jobs from the UI. For an API‑only host that
delegates compute elsewhere, `t3.small` is enough.

```bash
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \                # Ubuntu 22.04 LTS, us-east-1
  --instance-type t3.large \
  --key-name your-keypair \
  --security-group-ids sg-yyyyyyyy \
  --subnet-id subnet-zzzzzzzz \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=perceptrome-web}]'
```

Allocate an Elastic IP and associate it with the instance so the address
survives reboots:

```bash
aws ec2 allocate-address --domain vpc
aws ec2 associate-address --instance-id i-xxxx --allocation-id eipalloc-xxxx
```

### 3.4 DNS

Point an `A` record (or `ALIAS` if you use Route 53 + ALB) for
`app.perceptrome.com` at the EC2 Elastic IP (or the ALB DNS name).

---

## 4. Bootstrap the EC2 host

SSH in:

```bash
ssh -i ~/.ssh/your-keypair.pem ubuntu@<elastic-ip>
```

Install Docker Engine + Compose plugin (Ubuntu 22.04):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git make

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker ubuntu
# log out and back in so the group change takes effect, or run: newgrp docker
```

Verify:

```bash
docker --version
docker compose version
```

---

## 5. Get the code onto the instance

Two options.

**Option A — clone from GitHub** (preferred for repeatable deploys):

```bash
git clone https://github.com/wselbyGPT/Perceptrome.git
cd Perceptrome
```

**Option B — rsync from your workstation**:

```bash
rsync -avz --exclude '.venv' --exclude 'node_modules' \
  --exclude 'perceptrome_web/client/dist' \
  ./ ubuntu@<elastic-ip>:/home/ubuntu/Perceptrome/
```

---

## 6. Configure `.env` for production

The production compose file reads `perceptrome_web/server/.env`. Generate it
with the interactive helper:

```bash
python3 setup_web.py
# choose: production (AWS)
```

`setup_web.py` will prompt for:

- **Domain** (e.g. `app.perceptrome.com`) and HTTPS yes/no.
- **RDS endpoint, port, db, user, password** — assemble into a SQLAlchemy URL.
- **Redis URL** (optional, for shared rate‑limiting across replicas).
- **SES SMTP** credentials and `MAIL_FROM_EMAIL` (or pick `console` to skip).
- **Session TTL hours** and **cookie domain**.
- **Bootstrap admin email + password** — used once at startup to create the
  first admin user. The default email is `admin@perceptrome.dev` (a
  publicly‑routable TLD; pydantic's `EmailStr` rejects `.local`).

The generated file looks like:

```
APP_ENV=production
DATABASE_URL=postgresql+psycopg://perceptrome:***@perceptrome-db.xxxx.us-east-1.rds.amazonaws.com:5432/perceptrome
CORS_ORIGINS=https://app.perceptrome.com
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
COOKIE_DOMAIN=app.perceptrome.com
SESSION_TTL_HOURS=168
ALLOW_SELF_REGISTER=false
BOOTSTRAP_ADMIN_EMAIL=admin@perceptrome.dev
BOOTSTRAP_ADMIN_PASSWORD=<long-random-string>
MAIL_PROVIDER=smtp
MAIL_FROM_EMAIL=no-reply@perceptrome.com
SMTP_HOST=email-smtp.us-east-1.amazonaws.com
SMTP_PORT=587
SMTP_USERNAME=AKIA...
SMTP_PASSWORD=...
SMTP_USE_TLS=true
EMAIL_VERIFICATION_BASE_URL=https://app.perceptrome.com/verify-email
PASSWORD_RESET_BASE_URL=https://app.perceptrome.com/reset-password
```

The file is in `.dockerignore` and `.gitignore` — never commit it.

If you do not want to install Python on the host just to run `setup_web.py`,
generate `.env` on your workstation and `scp` it into
`perceptrome_web/server/.env`.

---

## 7. Build and start the stack

From the repo root on the EC2 instance:

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

What happens:

1. `migrate` runs `alembic upgrade head` against RDS and exits 0.
2. `api` starts uvicorn with 2 workers on port 8000 inside the container.
   On first start its `_bootstrap_admin()` startup hook creates the
   bootstrap admin user (`must_change_password=True`).
3. `nginx` starts and proxies `:80` → `api:8000`.

Verify:

```bash
docker compose -f docker-compose.prod.yml ps
curl -s http://localhost/api/health
# {"ok":true,"service":"perceptrome-api"}
```

Then hit `http://<elastic-ip>/` from a browser, log in with the bootstrap
credentials, and immediately change the password.

---

## 8. TLS

Pick **one** of these.

### 8.1 Terminate TLS at an Application Load Balancer (recommended)

1. Request an ACM certificate for `app.perceptrome.com` in the same region.
2. Create an ALB with two listeners:
   - `:80` → redirect to `:443`
   - `:443` → forward to a target group containing your EC2 instance on
     port `80`
3. Set the target group health check path to `/api/health`.
4. Make sure `perceptrome-ec2-sg` allows port 80 from `perceptrome-alb-sg`
   only (not the world).

The current `nginx.conf` already trusts `X-Forwarded-Proto` from upstream
proxies, and uvicorn is started with `--proxy-headers --forwarded-allow-ips
"*"`, so the API will see the original `https` scheme and set
`Secure` cookies correctly.

### 8.2 Terminate TLS on the EC2 box with Let's Encrypt

If you don't want an ALB, mount certificates into the nginx container.

```bash
sudo apt-get install -y certbot
sudo certbot certonly --standalone -d app.perceptrome.com
sudo mkdir -p /home/ubuntu/Perceptrome/perceptrome_web/nginx/certs
sudo cp /etc/letsencrypt/live/app.perceptrome.com/fullchain.pem \
       /home/ubuntu/Perceptrome/perceptrome_web/nginx/certs/
sudo cp /etc/letsencrypt/live/app.perceptrome.com/privkey.pem \
       /home/ubuntu/Perceptrome/perceptrome_web/nginx/certs/
sudo chown -R ubuntu:ubuntu /home/ubuntu/Perceptrome/perceptrome_web/nginx/certs
```

Then edit `perceptrome_web/nginx/nginx.conf`: uncomment the `server { listen
443 ssl; ... }` block (the template at the bottom of the file), set
`server_name app.perceptrome.com`, and uncomment the `return 301
https://...` redirect in the port‑80 server. Restart nginx:

```bash
docker compose -f docker-compose.prod.yml restart nginx
```

Renewals: certbot's auto‑renew systemd timer runs in the host; add a
`--deploy-hook` that copies the new cert into the `nginx/certs/` directory
and `docker compose ... restart nginx`.

---

## 9. Email (SES)

If you set `MAIL_PROVIDER=smtp` for password reset / invitation flows:

1. Verify your sender domain (or at minimum `MAIL_FROM_EMAIL`) in SES.
2. While SES is in sandbox mode you can only send **to** verified
   addresses. Request production access from the SES console before
   inviting real users.
3. Generate **SMTP credentials** (not raw IAM keys) in the SES console
   and put them in `SMTP_USERNAME` / `SMTP_PASSWORD`.

To smoke‑test email delivery without SES, set `MAIL_PROVIDER=console` —
messages get printed to the API container logs (`docker compose logs api`).

---

## 10. Operations

### Logs

```bash
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f nginx
docker compose -f docker-compose.prod.yml logs migrate
```

For longer‑term retention forward the Docker logs to CloudWatch using the
`awslogs` log driver in `docker-compose.prod.yml`:

```yaml
  api:
    logging:
      driver: awslogs
      options:
        awslogs-group: /perceptrome/api
        awslogs-region: us-east-1
        awslogs-stream-prefix: api
```

The EC2 instance role needs `logs:CreateLogStream` and `logs:PutLogEvents`
on that log group.

### Updating to a new release

```bash
cd ~/Perceptrome
git fetch && git checkout <tag-or-sha>
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
# migrate will rerun alembic upgrade head; api will rolling-restart
```

### Backups

- **RDS**: enable automated backups (`--backup-retention-period 7`) and
  consider a daily manual snapshot before any schema migration in
  production.
- **Object data** (run artifacts): if you enable S3 for artifact storage,
  rely on bucket versioning + lifecycle rules. Otherwise back up the EC2
  EBS volume on a schedule (Data Lifecycle Manager).

### Rotating the bootstrap admin password

The bootstrap user is created exactly once. To rotate via the database:

```bash
docker compose -f docker-compose.prod.yml exec api python - <<'PY'
from perceptrome_web.server.app.db.session import SessionLocal
from perceptrome_web.server.app.models.users import User
from perceptrome_web.server.app.core.security import hash_password
from sqlalchemy import select

db = SessionLocal()
u = db.execute(select(User).where(User.email == 'admin@perceptrome.dev')).scalar_one()
u.password_hash = hash_password('new-strong-password')
u.must_change_password = True
db.commit()
print('rotated')
PY
```

In normal use, prefer the in‑app password change flow.

---

## 11. Hardening checklist

- [ ] SSH restricted to a known CIDR or replaced with SSM Session Manager.
- [ ] RDS in a private subnet, security group limited to the EC2 SG.
- [ ] `BOOTSTRAP_ADMIN_PASSWORD` rotated immediately after first login;
      `must_change_password=true` is enforced on first login already.
- [ ] `ALLOW_SELF_REGISTER=false` (the default) so only invited users get in.
- [ ] `COOKIE_SECURE=true` and `COOKIE_DOMAIN` set to the public hostname.
- [ ] TLS terminated (ALB or nginx). Cookies will not work over plain HTTP
      because of `Secure`.
- [ ] CloudWatch alarms on EC2 CPU, RDS CPU/connections, ALB 5xx.
- [ ] EBS + RDS encryption at rest enabled (default for new RDS instances).

---

## 12. Troubleshooting

**`migrate` exits non‑zero, alembic can't connect**
The compose file passes `DATABASE_URL` straight from `.env`. Verify from
inside the API container:

```bash
docker compose -f docker-compose.prod.yml run --rm api \
  python -c "from perceptrome_web.server.app.core.config import settings; print(settings.database_url)"
```

then `psql "$DATABASE_URL"` to confirm the EC2 SG can actually reach RDS.

**Login returns `422 ... value is not a valid email address`**
Pydantic's `EmailStr` rejects reserved TLDs like `.local`, `.test`,
`.example`. Use a real public TLD for the bootstrap admin (the default is
`admin@perceptrome.dev`).

**Cookies don't persist over HTTPS**
Make sure (a) the request actually arrives at the API as `https` — check
that ALB listener forwards `X-Forwarded-Proto`, and (b) `COOKIE_DOMAIN`
matches the hostname in the browser bar exactly (no leading dot needed for
single host; add a leading dot only if you also serve subdomains).

**WebSocket disconnects after ~60s**
The bundled `nginx.conf` already sets `proxy_read_timeout 3600s` on `/ws`.
If you front nginx with an ALB, raise the ALB **idle timeout** from the
default 60 s to at least 300 s, otherwise long‑running training streams
will be killed.

**`docker compose build` runs out of memory on `t3.small`**
The Vite build step is the heavy one. Either bump the instance to
`t3.medium` for the build only, or build the image elsewhere and push it to
ECR, then `docker compose pull` on the EC2 host.

---

## 13. Going beyond a single node

When one EC2 box stops being enough:

- Push the image to **ECR** and run multiple API instances behind the ALB
  target group (the API is stateless apart from the DB and rate‑limit
  store; set `REDIS_URL` so login‑attempt counters are shared).
- Move long‑running compute (training, plasmid generation) onto worker
  instances or **AWS Batch** + **GPU** instance types, with the API only
  enqueuing jobs.
- Move artifacts to **S3** and serve via CloudFront.
- Promote RDS to Multi‑AZ.

These are out of scope for this single‑node guide, but the codebase is
already structured around them: `perceptrome/jobs/engine.py` is the
seam where worker dispatch would plug in.
