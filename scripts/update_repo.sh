#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/perceptrome}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
APP_SERVICE="${APP_SERVICE:-perceptrome}"
NGINX_SERVICE="${NGINX_SERVICE:-nginx}"
NGINX_RELOAD_SCRIPT="${NGINX_RELOAD_SCRIPT:-$REPO_DIR/tools/reload_nginx.sh}"

# Allow overrides: ./update_repo.sh [branch]
if [[ "${1:-}" != "" ]]; then
  BRANCH="$1"
fi

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }

service_exists() {
  local svc="$1"
  local svc_unit="${svc%.service}.service"

  if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-units --type=service --all 2>/dev/null | awk '{print $1}' | grep -Fxq "$svc"; then
      return 0
    fi
    if systemctl list-units --type=service --all 2>/dev/null | awk '{print $1}' | grep -Fxq "$svc_unit"; then
      return 0
    fi
    if systemctl list-unit-files --type=service 2>/dev/null | awk '{print $1}' | grep -Fxq "$svc"; then
      return 0
    fi
    if systemctl list-unit-files --type=service 2>/dev/null | awk '{print $1}' | grep -Fxq "$svc_unit"; then
      return 0
    fi
  fi

  if command -v service >/dev/null 2>&1; then
    if service --status-all 2>/dev/null | awk '{print $4}' | grep -Fxq "$svc"; then
      return 0
    fi
    if service --status-all 2>/dev/null | awk '{print $4}' | grep -Fxq "$svc_unit"; then
      return 0
    fi
  fi

  return 1
}

restart_service_if_present() {
  local svc="$1"

  if [[ -z "$svc" ]]; then
    return 0
  fi

  if ! service_exists "$svc"; then
    log "Service '$svc' not found; skipping restart."
    return 0
  fi

  log "Restarting service: $svc"
  if command -v systemctl >/dev/null 2>&1; then
    if sudo systemctl restart "$svc"; then
      log "Service restarted: $svc"
      return 0
    fi
    if sudo systemctl restart "${svc%.service}.service"; then
      log "Service restarted: ${svc%.service}.service"
      return 0
    fi
  fi

  if command -v service >/dev/null 2>&1; then
    if sudo service "$svc" restart; then
      log "Service restarted: $svc"
      return 0
    fi
    if sudo service "${svc%.service}" restart; then
      log "Service restarted: ${svc%.service}"
      return 0
    fi
  fi

  log "WARNING: Failed to restart service '$svc'"
  return 1
}

cd "$REPO_DIR"

# Ensure SSH agent is running with a loaded key so we aren't prompted repeatedly.
ensure_ssh_agent() {
  if ssh-add -l >/dev/null 2>&1; then
    return
  fi

  if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    log "Starting ssh-agent..."
    # shellcheck disable=SC2046
    eval $(ssh-agent -s)
  fi

  DEFAULT_KEY="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
  if [[ -f "$DEFAULT_KEY" ]]; then
    log "Loading SSH key: $DEFAULT_KEY"
    ssh-add "$DEFAULT_KEY" >/dev/null
  else
    log "SSH key not found at $DEFAULT_KEY; skipping ssh-add."
  fi
}

reload_web_stack() {
  local had_issue=0

  if [[ -n "$APP_SERVICE" ]]; then
    if ! restart_service_if_present "$APP_SERVICE"; then
      had_issue=1
    fi
  fi

  if [[ -x "$NGINX_RELOAD_SCRIPT" ]]; then
    log "Reloading nginx via $NGINX_RELOAD_SCRIPT"
    if "$NGINX_RELOAD_SCRIPT"; then
      log "nginx reload complete."
    else
      log "WARNING: nginx reload script failed."
      had_issue=1
    fi
  elif [[ -n "$NGINX_SERVICE" ]] && service_exists "$NGINX_SERVICE"; then
    log "Reloading nginx service: $NGINX_SERVICE"
    if command -v systemctl >/dev/null 2>&1; then
      if sudo systemctl reload "$NGINX_SERVICE" || sudo systemctl restart "$NGINX_SERVICE"; then
        log "nginx reload complete."
      else
        log "WARNING: nginx reload failed via systemd."
        had_issue=1
      fi
    elif command -v service >/dev/null 2>&1; then
      if sudo service "$NGINX_SERVICE" reload || sudo service "$NGINX_SERVICE" restart; then
        log "nginx reload complete."
      else
        log "WARNING: nginx reload failed via service command."
        had_issue=1
      fi
    else
      log "WARNING: No system service manager found to reload nginx."
      had_issue=1
    fi
  else
    log "nginx reload skipped; service not found."
  fi

  return "$had_issue"
}

ensure_ssh_agent

if [[ ! -d .git ]]; then
  echo "ERROR: $REPO_DIR is not a git repository (.git missing)" >&2
  exit 1
fi

log "Repo:   $REPO_DIR"
log "Remote: $REMOTE"
log "Branch: $BRANCH"

# Helpful context
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD || true)"
log "Current HEAD: $CURRENT_BRANCH ($(git rev-parse --short HEAD))"

# Make sure remote exists
if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "ERROR: Remote '$REMOTE' not found. Check: git remote -v" >&2
  exit 1
fi

# Fetch latest
log "Fetching..."
git fetch --all --prune

# Ensure branch exists locally; create tracking branch if needed
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
@@ -65,25 +205,31 @@ if [[ -n "$(git status --porcelain)" ]]; then
  STASHED=1
fi

# Fast-forward only pull (safe)
log "Pulling (fast-forward only) from $REMOTE/$BRANCH..."
if ! git pull --ff-only "$REMOTE" "$BRANCH"; then
  echo "ERROR: Pull failed (not a fast-forward). Your local branch may have diverged." >&2
  echo "You can resolve manually with one of:" >&2
  echo "  git pull --rebase $REMOTE $BRANCH" >&2
  echo "  git reset --hard $REMOTE/$BRANCH   # DESTRUCTIVE" >&2
  exit 1
fi

# Re-apply stashed changes (if any)
if [[ "$STASHED" -eq 1 ]]; then
  log "Re-applying stashed local changes..."
  if ! git stash pop; then
    echo "WARNING: stash pop had conflicts. Resolve conflicts, then commit or stash again." >&2
    exit 2
  fi
fi

log "Update complete."
log "Now at: $(git rev-parse --short HEAD)"
git --no-pager log -1 --oneline

if reload_web_stack; then
  log "Web stack reload complete."
else
  log "WARNING: Web stack reload encountered issues (see logs above)."
fi
