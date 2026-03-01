#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/perceptrome}"
APP_USER="${APP_USER:-perceptrome}"
SERVICE_NAME="${SERVICE_NAME:-perceptrome-web}"
APP_PORT="${APP_PORT:-8000}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"
DOMAIN_NAME="${DOMAIN_NAME:-}"
WEB_ARGS="${WEB_ARGS:---host 127.0.0.1 --port ${APP_PORT}}"

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "This script must run as root (for apt/systemd/nginx changes)." >&2
    exit 1
  fi
}

ensure_packages() {
  log "Updating apt package index"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y

  local packages=(
    python3-venv
    python3-pip
    git
    nginx
    certbot
    python3-certbot-nginx
  )

  log "Installing required packages"
  apt-get install -y "${packages[@]}"
}

ensure_app_user() {
  if id -u "${APP_USER}" >/dev/null 2>&1; then
    log "User ${APP_USER} already exists"
  else
    log "Creating system user ${APP_USER}"
    useradd --system --create-home --shell /bin/bash "${APP_USER}"
  fi
}

ensure_repo_checkout() {
  mkdir -p "${APP_DIR}"

  if [[ -d "${APP_DIR}/.git" ]]; then
    log "Repository already exists in ${APP_DIR}; pulling latest ${REPO_BRANCH}"
    git -C "${APP_DIR}" fetch --all --prune
    git -C "${APP_DIR}" checkout "${REPO_BRANCH}"
    git -C "${APP_DIR}" pull --ff-only origin "${REPO_BRANCH}"
  else
    if [[ -z "${REPO_URL}" ]]; then
      echo "REPO_URL must be set when ${APP_DIR} does not already contain a git checkout." >&2
      exit 1
    fi
    log "Cloning ${REPO_URL} (${REPO_BRANCH}) into ${APP_DIR}"
    git clone --branch "${REPO_BRANCH}" --single-branch "${REPO_URL}" "${APP_DIR}"
  fi

  chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
}

ensure_python_environment() {
  local venv_dir="${APP_DIR}/.venv"

  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    log "Creating virtual environment at ${venv_dir}"
    sudo -u "${APP_USER}" python3 -m venv "${venv_dir}"
  else
    log "Virtual environment already exists at ${venv_dir}"
  fi

  log "Installing Python dependencies"
  sudo -u "${APP_USER}" "${venv_dir}/bin/pip" install --upgrade pip
  sudo -u "${APP_USER}" "${venv_dir}/bin/pip" install -r "${APP_DIR}/requirements.txt"
  sudo -u "${APP_USER}" "${venv_dir}/bin/pip" install -e "${APP_DIR}"
}

ensure_systemd_service() {
  local service_file="/etc/systemd/system/${SERVICE_NAME}.service"

  log "Writing systemd unit: ${service_file}"
  cat > "${service_file}" <<SERVICE
[Unit]
Description=Perceptrome Web Service
After=network.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=${APP_DIR}/.venv/bin
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/perceptrome_web/perceptrome_web_cli.py ${WEB_ARGS}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

  systemctl daemon-reload
  systemctl enable --now "${SERVICE_NAME}"
  systemctl restart "${SERVICE_NAME}"
}

ensure_nginx_site() {
  local nginx_site="/etc/nginx/sites-available/${SERVICE_NAME}"
  local server_name="${DOMAIN_NAME:-_}"

  log "Writing nginx site config: ${nginx_site}"
  cat > "${nginx_site}" <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name ${server_name};

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX

  ln -sfn "${nginx_site}" "/etc/nginx/sites-enabled/${SERVICE_NAME}"
  rm -f /etc/nginx/sites-enabled/default

  nginx -t
  systemctl enable --now nginx
  systemctl reload nginx
}

maybe_enable_tls() {
  if [[ -z "${DOMAIN_NAME}" ]]; then
    log "DOMAIN_NAME not provided; skipping certbot TLS provisioning"
    return
  fi

  if [[ -d "/etc/letsencrypt/live/${DOMAIN_NAME}" ]]; then
    log "Certificate already exists for ${DOMAIN_NAME}; skipping initial issuance"
    return
  fi

  log "Requesting certificate for ${DOMAIN_NAME} with certbot"
  certbot --nginx \
    --non-interactive \
    --agree-tos \
    --register-unsafely-without-email \
    -d "${DOMAIN_NAME}" \
    --redirect
}

main() {
  require_root
  ensure_packages
  ensure_app_user
  ensure_repo_checkout
  ensure_python_environment
  ensure_systemd_service
  ensure_nginx_site
  maybe_enable_tls

  log "Bootstrap complete"
  systemctl status "${SERVICE_NAME}" --no-pager || true
}

main "$@"
