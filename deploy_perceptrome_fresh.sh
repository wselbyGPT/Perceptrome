#!/usr/bin/env bash
set -Eeuo pipefail

########################################
# Perceptrome fresh-server deploy script
# Ubuntu + systemd backend + Nginx proxy
########################################

# ====== CONFIG (edit if needed) ======
DOMAIN="${DOMAIN:-perceptrome.com}"
WWW_DOMAIN="${WWW_DOMAIN:-www.perceptrome.com}"
REPO_URL="${REPO_URL:-https://github.com/wselbyGPT/Perceptrome.git}"
BRANCH="${BRANCH:-main}"

# Which user should own/work in the repo (usually ubuntu on EC2)
APP_USER="${APP_USER:-ubuntu}"
APP_HOME="$(eval echo "~${APP_USER}")"

REPO_DIR="${REPO_DIR:-${APP_HOME}/Perceptrome}"
CLIENT_REL_PATH="${CLIENT_REL_PATH:-perceptrome_web/client}"
CLIENT_DIR="${REPO_DIR}/${CLIENT_REL_PATH}"
BACKEND_REL_PATH="${BACKEND_REL_PATH:-perceptrome_web/perceptrome_web_cli.py}"

NGINX_SITE_NAME="${NGINX_SITE_NAME:-perceptrome}"
NGINX_SITE_FILE="/etc/nginx/sites-available/${NGINX_SITE_NAME}"
NGINX_SITE_LINK="/etc/nginx/sites-enabled/${NGINX_SITE_NAME}"
SERVICE_NAME="${SERVICE_NAME:-perceptrome-web}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
APP_PORT="${APP_PORT:-8000}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_ARGS="${WEB_ARGS:---host ${WEB_HOST} --port ${APP_PORT}}"

# ====== helpers ======
log()  { echo -e "\n\033[1;36m==>\033[0m $*"; }
warn() { echo -e "\n\033[1;33mWARN:\033[0m $*" >&2; }
err()  { echo -e "\n\033[1;31mERROR:\033[0m $*" >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }
}

require_sudo() {
  if ! sudo -n true 2>/dev/null; then
    log "Requesting sudo privileges..."
    sudo -v
  fi
}

on_error() {
  err "Deployment failed on line $1."
}
trap 'on_error $LINENO' ERR

# ====== sanity checks ======
if [[ "$(id -u)" -eq 0 ]]; then
  err "Run this script as a normal user (e.g. ubuntu), not as root."
  exit 1
fi

require_cmd bash
require_cmd sudo
require_cmd curl
require_cmd grep

require_sudo

# ====== apt deps ======
log "Installing system packages"
sudo apt-get update -y
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  gnupg \
  nginx \
  python3-venv \
  python3-pip

# ====== Node.js (Vite 7 requires newer Node) ======
need_node_install=false

if ! command -v node >/dev/null 2>&1; then
  warn "Node is not installed. Will install Node 22."
  need_node_install=true
else
  NODE_VER_RAW="$(node -v || true)"       # e.g. v18.19.1
  NODE_MAJOR="$(echo "$NODE_VER_RAW" | sed -E 's/^v([0-9]+).*/\1/')"
  if [[ -z "${NODE_MAJOR}" ]] || (( NODE_MAJOR < 20 )); then
    warn "Found Node ${NODE_VER_RAW}. Vite build needs newer Node. Will install Node 22."
    need_node_install=true
  else
    log "Node already installed: ${NODE_VER_RAW}"
  fi
fi

if [[ "${need_node_install}" == "true" ]]; then
  log "Installing Node.js 22.x (NodeSource)"
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
    | sudo tee /etc/apt/sources.list.d/nodesource.list >/dev/null

  sudo apt-get update -y
  sudo apt-get install -y nodejs

  log "Installed Node: $(node -v), npm: $(npm -v)"
fi

# ====== clone or update repo ======
log "Preparing repo at ${REPO_DIR}"

sudo -u "${APP_USER}" mkdir -p "${APP_HOME}"

if [[ -d "${REPO_DIR}/.git" ]]; then
  log "Repo exists; updating branch ${BRANCH}"
  sudo -u "${APP_USER}" git -C "${REPO_DIR}" fetch --all --prune
  sudo -u "${APP_USER}" git -C "${REPO_DIR}" checkout "${BRANCH}" || true
  sudo -u "${APP_USER}" git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
else
  if [[ -d "${REPO_DIR}" ]]; then
    warn "${REPO_DIR} exists but is not a git repo. Backing it up."
    TS="$(date +%Y%m%d_%H%M%S)"
    sudo mv "${REPO_DIR}" "${REPO_DIR}.bak_${TS}"
  fi

  log "Cloning ${REPO_URL} (branch: ${BRANCH})"
  sudo -u "${APP_USER}" git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
fi

# ====== verify frontend path ======
if [[ ! -d "${CLIENT_DIR}" ]]; then
  err "Frontend directory not found: ${CLIENT_DIR}"
  echo "Check CLIENT_REL_PATH. Current repo tree snippet:"
  find "${REPO_DIR}" -maxdepth 4 -type d | sed 's#^#  #'
  exit 1
fi

if [[ ! -f "${CLIENT_DIR}/package.json" ]]; then
  err "No package.json found in ${CLIENT_DIR}"
  exit 1
fi

# ====== build frontend ======
log "Installing npm deps and building frontend"
pushd "${CLIENT_DIR}" >/dev/null

# Use npm install for compatibility with fresh servers if lockfile may change.
sudo -u "${APP_USER}" npm install
sudo -u "${APP_USER}" npm run build

if [[ ! -f "${CLIENT_DIR}/dist/index.html" ]]; then
  err "Build completed but dist/index.html was not found"
  exit 1
fi

popd >/dev/null

# ====== python backend ======
if [[ ! -f "${REPO_DIR}/${BACKEND_REL_PATH}" ]]; then
  err "Backend entrypoint not found: ${REPO_DIR}/${BACKEND_REL_PATH}"
  exit 1
fi

log "Creating/updating Python virtual environment"
sudo -u "${APP_USER}" python3 -m venv "${REPO_DIR}/.venv"
sudo -u "${APP_USER}" "${REPO_DIR}/.venv/bin/pip" install --upgrade pip
sudo -u "${APP_USER}" "${REPO_DIR}/.venv/bin/pip" install -r "${REPO_DIR}/requirements.txt"
sudo -u "${APP_USER}" "${REPO_DIR}/.venv/bin/pip" install -e "${REPO_DIR}"

log "Writing systemd service: ${SERVICE_FILE}"
sudo tee "${SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Perceptrome Web Service
After=network.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${REPO_DIR}
Environment=PATH=${REPO_DIR}/.venv/bin
ExecStart=${REPO_DIR}/.venv/bin/python ${REPO_DIR}/${BACKEND_REL_PATH} ${WEB_ARGS}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl is-active --quiet "${SERVICE_NAME}" || {
  err "Systemd service ${SERVICE_NAME} is not active"
  sudo systemctl status "${SERVICE_NAME}" --no-pager || true
  exit 1
}

# ====== nginx site config ======
log "Writing Nginx site config: ${NGINX_SITE_FILE}"

sudo tee "${NGINX_SITE_FILE}" >/dev/null <<EOF
server {
    listen 80;
    listen [::]:80;

    server_name ${DOMAIN} ${WWW_DOMAIN};

    location /ws {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

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
EOF

# Enable custom site, disable default
if [[ ! -L "${NGINX_SITE_LINK}" ]]; then
  sudo ln -s "${NGINX_SITE_FILE}" "${NGINX_SITE_LINK}"
fi

if [[ -e /etc/nginx/sites-enabled/default ]]; then
  sudo rm -f /etc/nginx/sites-enabled/default
fi

# ====== validate + reload nginx ======
log "Testing and reloading Nginx"
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# ====== local smoke test ======
log "Running local smoke test"
BACKEND_STATUS="$(curl -s -o /tmp/perceptrome_backend_smoke.html -w "%{http_code}" "http://127.0.0.1:${APP_PORT}/")" || true
echo "Backend HTTP status: ${BACKEND_STATUS}"

if [[ "${BACKEND_STATUS}" != "200" ]]; then
  warn "Expected backend 200 but got ${BACKEND_STATUS}. Showing first lines of response:"
  head -40 /tmp/perceptrome_backend_smoke.html || true
  exit 1
fi

HTTP_STATUS="$(curl -s -o /tmp/perceptrome_smoke.html -w "%{http_code}" http://127.0.0.1 -H "Host: ${DOMAIN}")" || true
echo "Nginx HTTP status: ${HTTP_STATUS}"

if [[ "${HTTP_STATUS}" != "200" ]]; then
  warn "Expected 200 but got ${HTTP_STATUS}. Showing first lines of response:"
  head -40 /tmp/perceptrome_smoke.html || true
  exit 1
fi

echo
echo "✅ Deploy complete"
echo "Domain: http://${DOMAIN}"
echo "Repo: ${REPO_DIR}"
echo "Backend: ${SERVICE_NAME} (127.0.0.1:${APP_PORT})"
echo
echo "Next recommended step (HTTPS):"
echo "  sudo apt-get install -y certbot python3-certbot-nginx"
echo "  sudo certbot --nginx -d ${DOMAIN} -d ${WWW_DOMAIN}"
