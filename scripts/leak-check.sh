#!/usr/bin/env bash
# leak-check.sh — deterministic, dependency-free leak guard (the hard gate).
#
# Scans for GENERIC leak tells only (no domain wordlist): real local user paths,
# hardcoded secrets/keys, non-example emails, and DO-NOT-SHIP markers.
#
# Usage:
#   scripts/leak-check.sh            # scan the STAGED diff (default; for pre-commit)
#   scripts/leak-check.sh PATH       # scan a file or directory recursively
#
# Exit 0 + "clean" line if nothing found; non-zero + file:line of each hit otherwise.
# Requires only POSIX tools + grep + git. No network, no Python.

set -u

# --- temp workspace ----------------------------------------------------------
WORK=$(mktemp -d)
HITS="$WORK/hits"
: >"$HITS"
trap 'rm -rf "$WORK"' EXIT

# --- what we never scan (so the pattern list can't self-trip, plus noise) ----
is_excluded() {
  case "$1" in
    # The guard machinery itself — these files necessarily CONTAIN the pattern
    # vocabulary (regexes, category names, the DO-NOT-SHIP token) and would self-trip.
    */scripts/leak-check.sh|scripts/leak-check.sh|./scripts/leak-check.sh)        return 0 ;;
    */scripts/semantic-audit.py|scripts/semantic-audit.py)                       return 0 ;;
    */policies/generic.pack.yaml|policies/generic.pack.yaml)                     return 0 ;;
    */.git/*|.git/*|*/.venv/*|.venv/*|*/__pycache__/*|__pycache__/*)       return 0 ;;
    */.pytest_cache/*|.pytest_cache/*|*.egg-info/*)                        return 0 ;;
    # The SYNTHETIC sample corpus and the redaction/PII test suites deliberately
    # plant FAKE secrets/credentials/emails to exercise the redaction pipeline
    # (see redact.py / README "Secret redaction"). They are intended fixtures, not
    # leaks — never gate on them. (Reserved TLDs .example/.test are also allowed below.)
    */data/sample/*|data/sample/*)                                         return 0 ;;
    */tests/*|tests/*)                                                     return 0 ;;
  esac
  return 1
}

# --- pick a grep that understands PCRE inline flags --------------------------
if printf 'x' | grep -qP 'x' 2>/dev/null; then
  GREP_PCRE=1            # grep -P available: keep (?i) inline flags
else
  GREP_PCRE=0            # fall back to -E + global -i, with (?i) stripped
fi

# --- the GENERIC pattern set (no domain terms) -------------------------------
# Format: "LABEL<TAB>regex". (?i) means case-insensitive for that pattern.
PATTERNS=$(printf '%s\n' \
'local-user-path	/Users/[A-Za-z0-9._-]+/' \
'local-user-path	/home/[A-Za-z0-9._-]+/' \
'windows-user-path	[Cc]:\\Users\\[^\\]+\\' \
'mounted-volume	/Volumes/[A-Za-z0-9 ._-]+/' \
'aws-access-key	(AKIA|ASIA)[0-9A-Z]{16}' \
'aws-secret-ref	(?i)aws_secret_access_key' \
'openai-key	sk-[A-Za-z0-9]{20,}' \
'anthropic-key	sk-ant-[A-Za-z0-9_-]{20,}' \
'github-token	gh[pousr]_[A-Za-z0-9]{30,}' \
'slack-token	xox[baprs]-[A-Za-z0-9-]{10,}' \
'google-api-key	AIza[0-9A-Za-z_-]{35}' \
'private-key-block	-----BEGIN ([A-Z]+ )?PRIVATE KEY-----' \
'generic-secret-assign	(?i)(api[_-]?key|secret|passwd|password|token|bearer)[[:space:]]*[:=][[:space:]]*["'"'"']?[A-Za-z0-9_./+-]{16,}' \
'db-creds-url	(postgres|postgresql|mysql|mongodb|redis)://[^[:space:]"'"'"']*:[^[:space:]"'"'"'@]+@' \
'do-not-ship	(?i)do[[:space:]_-]*not[[:space:]_-]*(ship|commit)' \
)

EMAIL_RE='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'

# scan_one CONTENT_FILE DISPLAY_NAME — appends "DISPLAY:matchline" to $HITS.
scan_one() {
  content_file="$1"; display="$2"; label=""; re=""; rendered=""; line=""

  printf '%s\n' "$PATTERNS" | while IFS=$'\t' read -r label re; do
    [ -z "$label" ] && continue
    if [ "$GREP_PCRE" -eq 1 ]; then
      grep -nP -- "$re" "$content_file" 2>/dev/null
    else
      rendered=$(printf '%s' "$re" | sed 's/(?i)//g')
      grep -niE -- "$rendered" "$content_file" 2>/dev/null
    fi | while IFS= read -r line; do
      printf 'LEAK[%s] %s:%s\n' "$label" "$display" "$line" >>"$HITS"
    done
  done

  # Emails: flag only if a non-example/non-placeholder address appears on the line.
  grep -nE -- "$EMAIL_RE" "$content_file" 2>/dev/null | while IFS= read -r line; do
    bad=$(printf '%s\n' "$line" | grep -oE -- "$EMAIL_RE" \
            | grep -ivE '@([A-Za-z0-9.-]*\.)?(example|test|invalid|localhost)(\.[a-z]+)?$|@yourdomain\.[a-z]+$' || true)
    if [ -n "$bad" ]; then
      printf 'LEAK[non-example-email] %s:%s\n' "$display" "$line" >>"$HITS"
    fi
  done
}

# --- gather targets ----------------------------------------------------------
if [ "$#" -ge 1 ]; then
  TARGET="$1"
  if [ -d "$TARGET" ]; then
    while IFS= read -r f; do
      is_excluded "$f" && continue
      grep -Iq . "$f" 2>/dev/null && scan_one "$f" "$f"
    done < <(find "$TARGET" -type f 2>/dev/null)
  elif [ -f "$TARGET" ]; then
    is_excluded "$TARGET" || { grep -Iq . "$TARGET" 2>/dev/null && scan_one "$TARGET" "$TARGET"; }
  else
    echo "leak-check: no such path: $TARGET" >&2
    exit 2
  fi
else
  files=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null)
  if [ -z "$files" ]; then
    echo "leak-check: clean (no staged files to scan)"
    exit 0
  fi
  printf '%s\n' "$files" | while IFS= read -r f; do
    [ -z "$f" ] && continue
    is_excluded "$f" && continue
    tmp="$WORK/staged.blob"
    if git show ":$f" >"$tmp" 2>/dev/null && grep -Iq . "$tmp" 2>/dev/null; then
      scan_one "$tmp" "$f"
    fi
  done
fi

# --- verdict -----------------------------------------------------------------
if [ -s "$HITS" ]; then
  cat "$HITS"
  echo ""
  echo "leak-check: FAIL — potential leaks found above. Remove/redact before committing."
  exit 1
fi

echo "leak-check: clean (no generic leak patterns found)"
exit 0
