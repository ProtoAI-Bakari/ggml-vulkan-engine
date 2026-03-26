#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# mount_shared.sh — Mount AGENT dir from sys1 to MLX nodes via SSHFS
# ═══════════════════════════════════════════════════════════════════════════════
# STATUS: EXPERIMENTAL
#
# Mounts ~/AGENT from sys1 (10.255.255.128) as read-only ~/AGENT_SHARED on
# each MLX node (sys2-sys7). Agents on those nodes can read project files,
# task queues, and configs without SCP copies going stale.
#
# Prerequisites:
#   - sshfs + fuse3 installed on each MLX node (script installs if missing)
#   - sshpass installed on this machine (sys1)
#   - SSH access from MLX nodes back to sys1 (z@10.255.255.128)
#   - ~/DEV/authpass contains the SSH password
#
# Usage:
#   ./mount_shared.sh              # Mount on all MLX nodes (sys2-sys7)
#   ./mount_shared.sh sys3         # Mount on sys3 only
#   ./mount_shared.sh --unmount    # Unmount from all nodes
#   ./mount_shared.sh --status     # Check mount status on all nodes
#   ./mount_shared.sh --test sys4  # Test on one node only
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

SYS1_IP="10.255.255.128"
SYS1_USER="z"
REMOTE_PATH="/home/z/AGENT"
MOUNT_POINT="AGENT_SHARED"  # relative to home dir on each node

SSH_USER="z"
PASSFILE="$HOME/DEV/authpass"
SSH_OPTS="-o ConnectTimeout=8 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30"

# MLX node map: name -> IP
declare -A NODES=(
    [sys2]="10.255.255.2"
    [sys3]="10.255.255.3"
    [sys4]="10.255.255.4"
    [sys5]="10.255.255.5"
    [sys6]="10.255.255.6"
    [sys7]="10.255.255.7"
)

# ── Colors ────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
CYN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RST='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────

log()  { echo -e "${CYN}[mount]${RST} $*"; }
ok()   { echo -e "${GRN}  [OK]${RST} $*"; }
warn() { echo -e "${YLW}[WARN]${RST} $*"; }
err()  { echo -e "${RED} [ERR]${RST} $*"; }

die() { err "$*"; exit 1; }

# Validate passfile exists
[[ -f "$PASSFILE" ]] || die "Password file not found: $PASSFILE"

# Run command on a remote node via sshpass + ssh
remote_exec() {
    local ip="$1"
    shift
    sshpass -f "$PASSFILE" ssh $SSH_OPTS -T "${SSH_USER}@${ip}" "$@" 2>/dev/null
}

# Run command on a remote node that needs sudo
remote_sudo() {
    local ip="$1"
    shift
    local pass
    pass=$(cat "$PASSFILE")
    sshpass -f "$PASSFILE" ssh $SSH_OPTS -T "${SSH_USER}@${ip}" \
        "echo '${pass}' | sudo -S bash -c '$*'" 2>/dev/null
}

# ── Per-node operations ──────────────────────────────────────────────────────

ensure_sshfs_installed() {
    local name="$1" ip="$2"

    # Check if sshfs is already available
    if remote_exec "$ip" "which sshfs" &>/dev/null; then
        return 0
    fi

    log "Installing sshfs on ${name} (${ip})..."

    # Try dnf first (Fedora/Asahi), then pacman, then apt
    if remote_sudo "$ip" "dnf install -y fuse-sshfs 2>/dev/null || pacman -S --noconfirm sshfs 2>/dev/null || apt-get install -y sshfs 2>/dev/null"; then
        if remote_exec "$ip" "which sshfs" &>/dev/null; then
            ok "sshfs installed on ${name}"
            return 0
        fi
    fi

    err "Failed to install sshfs on ${name} -- install manually: sudo dnf install fuse-sshfs"
    return 1
}

mount_node() {
    local name="$1" ip="$2"

    log "${BOLD}${name}${RST} (${ip}) — mounting..."

    # 1) Check node is reachable
    if ! remote_exec "$ip" "echo ok" &>/dev/null; then
        err "${name}: unreachable"
        return 1
    fi

    # 2) Check if already mounted
    if remote_exec "$ip" "mountpoint -q ~/\${MOUNT_POINT:-$MOUNT_POINT} 2>/dev/null && echo mounted" 2>/dev/null | grep -q mounted; then
        ok "${name}: already mounted"
        return 0
    fi

    # 3) Ensure sshfs is installed
    if ! ensure_sshfs_installed "$name" "$ip"; then
        return 1
    fi

    # 4) Create mount point
    remote_exec "$ip" "mkdir -p ~/${MOUNT_POINT}"

    # 5) Copy the password file to the node temporarily for sshfs auth
    #    sshfs on the remote node needs to auth back to sys1
    sshpass -f "$PASSFILE" scp $SSH_OPTS "$PASSFILE" "${SSH_USER}@${ip}:/tmp/.mount_auth" 2>/dev/null

    # 6) Mount via sshfs (read-only, with reconnect and caching)
    #    The remote node mounts sys1's AGENT dir via sshfs
    local mount_cmd="sshpass -f /tmp/.mount_auth sshfs \
        -o ro \
        -o StrictHostKeyChecking=no \
        -o reconnect \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o cache=yes \
        -o cache_timeout=120 \
        -o attr_timeout=120 \
        -o entry_timeout=120 \
        -o allow_other \
        ${SYS1_USER}@${SYS1_IP}:${REMOTE_PATH} ~/${MOUNT_POINT} 2>&1"

    local result
    result=$(remote_exec "$ip" "$mount_cmd" 2>&1) || true

    # 7) Clean up temp auth file
    remote_exec "$ip" "rm -f /tmp/.mount_auth" 2>/dev/null || true

    # 8) Verify mount succeeded
    if remote_exec "$ip" "mountpoint -q ~/${MOUNT_POINT} 2>/dev/null && ls ~/${MOUNT_POINT}/GO_PROMPT.md" &>/dev/null; then
        ok "${name}: mounted ${SYS1_IP}:${REMOTE_PATH} -> ~/${MOUNT_POINT} (read-only)"
        return 0
    fi

    # allow_other might fail if /etc/fuse.conf doesn't have user_allow_other
    # Retry without allow_other
    warn "${name}: retrying without allow_other..."
    mount_cmd="sshpass -f /tmp/.mount_auth_retry sshfs \
        -o ro \
        -o StrictHostKeyChecking=no \
        -o reconnect \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o cache=yes \
        -o cache_timeout=120 \
        ${SYS1_USER}@${SYS1_IP}:${REMOTE_PATH} ~/${MOUNT_POINT} 2>&1"

    # Re-copy auth for retry
    sshpass -f "$PASSFILE" scp $SSH_OPTS "$PASSFILE" "${SSH_USER}@${ip}:/tmp/.mount_auth_retry" 2>/dev/null

    result=$(remote_exec "$ip" "$mount_cmd" 2>&1) || true
    remote_exec "$ip" "rm -f /tmp/.mount_auth_retry" 2>/dev/null || true

    if remote_exec "$ip" "mountpoint -q ~/${MOUNT_POINT} 2>/dev/null && ls ~/${MOUNT_POINT}/" &>/dev/null; then
        ok "${name}: mounted (without allow_other)"
        return 0
    fi

    # If sshfs also needs sshpass on remote node
    if ! remote_exec "$ip" "which sshpass" &>/dev/null; then
        warn "${name}: sshpass not found on remote — installing..."
        remote_sudo "$ip" "dnf install -y sshpass 2>/dev/null || apt-get install -y sshpass 2>/dev/null" || true
    fi

    err "${name}: mount failed. Output: ${result}"
    return 1
}

unmount_node() {
    local name="$1" ip="$2"

    log "${name} (${ip}) — unmounting..."

    if ! remote_exec "$ip" "echo ok" &>/dev/null; then
        err "${name}: unreachable"
        return 1
    fi

    if ! remote_exec "$ip" "mountpoint -q ~/${MOUNT_POINT} 2>/dev/null && echo mounted" 2>/dev/null | grep -q mounted; then
        warn "${name}: not mounted"
        return 0
    fi

    if remote_exec "$ip" "fusermount3 -u ~/${MOUNT_POINT} 2>/dev/null || fusermount -u ~/${MOUNT_POINT} 2>/dev/null || umount ~/${MOUNT_POINT} 2>/dev/null"; then
        ok "${name}: unmounted"
        return 0
    fi

    # Force unmount if graceful fails
    warn "${name}: trying lazy unmount..."
    remote_sudo "$ip" "umount -l /home/${SSH_USER}/${MOUNT_POINT}" 2>/dev/null || true
    ok "${name}: lazy unmounted"
}

status_node() {
    local name="$1" ip="$2"

    if ! remote_exec "$ip" "echo ok" &>/dev/null; then
        echo -e "  ${RED}${name}${RST} (${ip}): ${RED}UNREACHABLE${RST}"
        return
    fi

    local mounted="no"
    local sshfs_installed="no"
    local file_count="-"

    if remote_exec "$ip" "which sshfs" &>/dev/null; then
        sshfs_installed="yes"
    fi

    if remote_exec "$ip" "mountpoint -q ~/${MOUNT_POINT}" &>/dev/null; then
        mounted="yes"
        file_count=$(remote_exec "$ip" "ls ~/${MOUNT_POINT}/ 2>/dev/null | wc -l" 2>/dev/null || echo "?")
    fi

    if [[ "$mounted" == "yes" ]]; then
        echo -e "  ${GRN}${name}${RST} (${ip}): ${GRN}MOUNTED${RST} — ${file_count} items in ~/${MOUNT_POINT}"
    else
        local has_dir="no"
        if remote_exec "$ip" "test -d ~/${MOUNT_POINT}" &>/dev/null; then
            has_dir="yes"
        fi
        echo -e "  ${YLW}${name}${RST} (${ip}): ${YLW}NOT MOUNTED${RST} — sshfs=${sshfs_installed}, dir=${has_dir}"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══ AGENT Shared Mount (EXPERIMENTAL) ═══${RST}"
echo -e "${DIM}Source: ${SYS1_USER}@${SYS1_IP}:${REMOTE_PATH}${RST}"
echo -e "${DIM}Target: ~/${MOUNT_POINT} (read-only via sshfs)${RST}"
echo ""

ACTION="mount"
TARGET_NODES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --unmount|--umount|-u)
            ACTION="unmount"
            shift
            ;;
        --status|-s)
            ACTION="status"
            shift
            ;;
        --test|-t)
            ACTION="test"
            shift
            if [[ $# -gt 0 ]]; then
                TARGET_NODES+=("$1")
                shift
            fi
            ;;
        sys[2-7])
            TARGET_NODES+=("$1")
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--unmount|--status|--test NODE|sys2..sys7]"
            exit 0
            ;;
        *)
            warn "Unknown argument: $1"
            shift
            ;;
    esac
done

# Default to all nodes if none specified
if [[ ${#TARGET_NODES[@]} -eq 0 ]]; then
    TARGET_NODES=(sys2 sys3 sys4 sys5 sys6 sys7)
fi

# Execute
SUCCESS=0
FAIL=0

for name in "${TARGET_NODES[@]}"; do
    ip="${NODES[$name]:-}"
    if [[ -z "$ip" ]]; then
        err "Unknown node: ${name}"
        ((FAIL++))
        continue
    fi

    case "$ACTION" in
        mount|test)
            if mount_node "$name" "$ip"; then
                ((SUCCESS++))
            else
                ((FAIL++))
            fi
            ;;
        unmount)
            if unmount_node "$name" "$ip"; then
                ((SUCCESS++))
            else
                ((FAIL++))
            fi
            ;;
        status)
            status_node "$name" "$ip"
            ((SUCCESS++))
            ;;
    esac
done

# Summary
echo ""
echo -e "${BOLD}─── Summary ───${RST}"
if [[ "$ACTION" == "status" ]]; then
    echo -e "  Checked: ${SUCCESS} nodes"
else
    echo -e "  Success: ${GRN}${SUCCESS}${RST}  Failed: ${RED}${FAIL}${RST}"
fi

if [[ "$ACTION" == "test" && $SUCCESS -gt 0 ]]; then
    echo ""
    echo -e "${CYN}Test passed.${RST} Run without --test to mount all nodes."
fi

echo ""
exit $FAIL
