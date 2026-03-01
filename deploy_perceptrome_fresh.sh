#!/usr/bin/env bash
set -Eeuo pipefail

########################################
# Perceptrome fresh-server deploy script
# Ubuntu + Nginx + Vite static frontend
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

WEB_ROOT="${WEB_ROOT:-/var/www/perceptrome}"
NGINX_SITE_NAME="${NGINX_SITE_NAME:-perceptrome}"
NGINX_SITE_FILE="/etc/nginx/sites-available/${NGINX_SITE_NAME}"
NGINX_SITE_LINK="/etc/nginx/sites-enabled/${NGINX_SITE_NAME}"

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
  rsync \
  nginx

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

# ====== deploy static files ======
log "Deploying built files to ${WEB_ROOT}"
sudo mkdir -p "${WEB_ROOT}"
sudo rsync -a --delete "${CLIENT_DIR}/dist/" "${WEB_ROOT}/"

# Ensure readable by nginx
sudo find "${WEB_ROOT}" -type d -exec chmod 755 {} \;
sudo find "${WEB_ROOT}" -type f -exec chmod 644 {} \;

# ====== nginx site config ======
log "Writing Nginx site config: ${NGINX_SITE_FILE}"

sudo tee "${NGINX_SITE_FILE}" >/dev/null <<EOF
server {
    listen 80;
    listen [::]:80;

    server_name ${DOMAIN} ${WWW_DOMAIN};

    root ${WEB_ROOT};
    index index.html;

    # SPA routing (Vite/React)
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # Static hashed assets
    location /assets/ {
        try_files \$uri =404;
        expires 1y;
        add_header Cache-Control "public, immutable";
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
HTTP_STATUS="$(curl -s -o /tmp/perceptrome_smoke.html -w "%{http_code}" http://127.0.0.1 -H "Host: ${DOMAIN}")" || true
echo "HTTP status: ${HTTP_STATUS}"

if [[ "${HTTP_STATUS}" != "200" ]]; then
  warn "Expected 200 but got ${HTTP_STATUS}. Showing first lines of response:"
  head -40 /tmp/perceptrome_smoke.html || true
  exit 1
fi

echo
echo "✅ Deploy complete"
echo "Domain: http://${DOMAIN}"
echo "Web root: ${WEB_ROOT}"
echo "Repo: ${REPO_DIR}"
echo
echo "Next recommended step (HTTPS):"
echo "  sudo apt-get install -y certbot python3-certbot-nginx"
echo "  sudo certbot --nginx -d ${DOMAIN} -d ${WWW_DOMAIN}"
