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
  :
else
  if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
    log "Creating local branch '$BRANCH' tracking '$REMOTE/$BRANCH'"
    git switch -c "$BRANCH" --track "$REMOTE/$BRANCH"
  else
    echo "ERROR: Branch '$BRANCH' not found on remote '$REMOTE'." >&2
    echo "Try: git branch -r" >&2
    exit 1
  fi
fi

# Switch to branch if not already
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  log "Switching to $BRANCH"
  git switch "$BRANCH"
fi

# Stash local changes if any
STASHED=0
if [[ -n "$(git status --porcelain)" ]]; then
  log "Local changes detected; stashing before pull..."
  git stash push -u -m "auto-stash before update ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
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
