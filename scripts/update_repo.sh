#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/perceptrome}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"

# Allow overrides: ./update_repo.sh [branch]
if [[ "${1:-}" != "" ]]; then
  BRANCH="$1"
fi

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }

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
