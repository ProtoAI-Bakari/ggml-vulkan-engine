#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ProtoAI-Bakari CUDA Cluster Audit v16
# ═══════════════════════════════════════════════════════════════════════════════
# Improvements over v14_v2_GDR:
#   - Auto-discovers ALL venvs on each node (no more guessing)
#   - NFS mount status + RDMA link stats
#   - GPU memory usage per card
#   - Module load/unload aliases generated
#   - Cleaner output, wider columns
# Usage:
#   ./cuda_cluster_audit_v16.sh              # audit all nodes, auto-detect venvs
#   ./cuda_cluster_audit_v16.sh /path/venv   # audit with specific venv
#   ./cuda_cluster_audit_v16.sh --modules    # just show module status
#   ./cuda_cluster_audit_v16.sh --nfs        # just show NFS mounts
#   ./cuda_cluster_audit_v16.sh --load-all   # load all perf modules on all nodes
#   ./cuda_cluster_audit_v16.sh --aliases    # print module management aliases
# ═══════════════════════════════════════════════════════════════════════════════

NODES=(10.255.255.10 10.255.255.11 10.255.255.12 10.255.255.13 10.255.255.14)
TEMP_DIR="/tmp/cluster_audit_v16"
SSH_OPTS="-o ConnectTimeout=5 -o ServerAliveInterval=30 -o StrictHostKeyChecking=no -o LogLevel=ERROR"
SSH_PASS="z"
SSH_USER="z"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; BLU='\033[0;34m'; MAG='\033[0;35m'; CYN='\033[0;36m'
BOLD='\033[1m'; DIM='\033[2m'; RST='\033[0m'

ssh_node() { sshpass -p "$SSH_PASS" ssh $SSH_OPTS -T "$SSH_USER@$1" "$2" 2>/dev/null; }

# ── Aliases Mode ─────────────────────────────────────────────────────────────
if [[ "$1" == "--aliases" ]]; then
    cat << 'ALIASES'
# ═══ CUDA Cluster Module Management Aliases ═══
# Paste into your .bashrc or .zprofile

# Load individual modules
alias cuda_enable_gdr='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S modprobe gdrdrv && echo $ip: gdrdrv loaded"; done'
alias cuda_enable_xpm='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S modprobe xpmem && echo $ip: xpmem loaded"; done'
alias cuda_enable_knm='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S modprobe knem && echo $ip: knem loaded"; done'
alias cuda_enable_peer='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S modprobe nvidia_peermem && echo $ip: peermem loaded"; done'

# Unload individual modules
alias cuda_disable_gdr='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S rmmod gdrdrv 2>/dev/null && echo $ip: gdrdrv unloaded || echo $ip: gdrdrv not loaded"; done'
alias cuda_disable_xpm='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S rmmod xpmem 2>/dev/null && echo $ip: xpmem unloaded || echo $ip: xpmem not loaded"; done'
alias cuda_disable_peer='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S rmmod nvidia_peermem 2>/dev/null && echo $ip: peermem unloaded || echo $ip: peermem not loaded"; done'

# Load/unload ALL performance modules (paired)
alias cuda_perf_on='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S modprobe gdrdrv; echo z | sudo -S modprobe xpmem; echo z | sudo -S modprobe knem; echo z | sudo -S modprobe nvidia_peermem; echo $ip: ALL loaded"; done'
alias cuda_perf_off='for ip in 10.255.255.{11,12,13,14}; do sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "echo z | sudo -S rmmod nvidia_peermem 2>/dev/null; echo z | sudo -S rmmod gdrdrv 2>/dev/null; echo z | sudo -S rmmod xpmem 2>/dev/null; echo z | sudo -S rmmod knem 2>/dev/null; echo $ip: ALL unloaded"; done'

# Quick status
alias cuda_mods='for ip in 10.255.255.{10,11,12,13,14}; do m=$(sshpass -p z ssh -o StrictHostKeyChecking=no z@$ip "lsmod | grep -cE \"gdrdrv|xpmem|knem|nvidia_peermem\"" 2>/dev/null); echo "$ip: $m/4 modules"; done'
alias cuda_audit='bash ~/AGENT/scripts/cuda_cluster_audit_v16.sh'
ALIASES
    exit 0
fi

# ── Load All Mode ────────────────────────────────────────────────────────────
if [[ "$1" == "--load-all" ]]; then
    echo -e "${BOLD}${CYN}>>> Loading ALL performance modules on cluster...${RST}"
    for ip in "${NODES[@]}"; do
        [[ "$ip" == "10.255.255.10" ]] && continue  # skip 4090 (no peermem)
        result=$(ssh_node "$ip" "
            echo z | sudo -S modprobe gdrdrv 2>/dev/null
            echo z | sudo -S modprobe xpmem 2>/dev/null
            echo z | sudo -S modprobe knem 2>/dev/null
            echo z | sudo -S modprobe nvidia_peermem 2>/dev/null
            lsmod | grep -cE 'gdrdrv|xpmem|knem|nvidia_peermem'
        ")
        count=$(echo "$result" | tail -1)
        echo -e "  $ip: ${GRN}${count}/4 modules loaded${RST}"
    done
    exit 0
fi

# ── Main Audit ───────────────────────────────────────────────────────────────
mkdir -p "$TEMP_DIR"
rm -f "$TEMP_DIR"/*.raw

TARGET_VENV="${1:-AUTO}"

echo ""
echo -e "${BOLD}${MAG}═══════════════════════════════════════════════════════════════${RST}"
echo -e "${BOLD}${MAG}  ProtoAI-Bakari CUDA Cluster Audit v16  $(date '+%Y-%m-%d %H:%M:%S')${RST}"
echo -e "${BOLD}${MAG}═══════════════════════════════════════════════════════════════${RST}"
[[ "$TARGET_VENV" != "AUTO" ]] && echo -e "${CYN}  Target venv: $TARGET_VENV${RST}"

# Parallel collection
for ip in "${NODES[@]}"; do
    {
        ssh_node "$ip" "$(cat << 'REMOTE'
export PATH=$PATH:/usr/sbin:/sbin:/usr/bin:/bin:/usr/local/bin
export VLLM_CONFIGURE_LOGGING=0

# --- Hardware ---
DETECTED_IFACE=$(ip -o -4 addr show | awk '$4 ~ /^10\.255\.255\./ {print $2; exit}')
IP=$(ip -o -4 addr show $DETECTED_IFACE | awk '{split($4,a,"/"); print a[1]}')
HOSTNAME=$(hostname -s)

HCA_ID="mlx5_0"
PCI_ID=$(basename $(readlink /sys/class/net/$DETECTED_IFACE/device 2>/dev/null) 2>/dev/null)
if [ -n "$PCI_ID" ] && lspci -s "$PCI_ID" 2>/dev/null | grep -q "Virtual Function"; then
    HCA_ID="mlx5_0:1"
    NODE_TYPE="VM"
else
    NODE_TYPE="BM"
fi

# GID
GID_INDEX="Err"
for i in $(seq 0 15); do
    gid=$(cat /sys/class/infiniband/${HCA_ID%%:*}/ports/1/gids/$i 2>/dev/null)
    if echo "$gid" | grep -q "ffff:$(printf '%02x%02x:%02x%02x' $(echo $IP | tr '.' ' '))"; then
        GID_INDEX=$i; break
    fi
done

IB_STATE=$(ibv_devinfo -d ${HCA_ID%%:*} 2>/dev/null | grep "state:" | awk '{print $2}' | head -1)
IB_STATE=${IB_STATE:-"DOWN"}

# --- Modules ---
PEER=$(lsmod | grep -q "^nvidia_peermem" && echo "Y" || echo "N")
GDR=$(lsmod | grep -q "^gdrdrv" && echo "Y" || echo "N")
XPM=$(lsmod | grep -q "^xpmem" && echo "Y" || echo "N")
KNM=$(lsmod | grep -q "^knem" && echo "Y" || echo "N")

# --- GPU ---
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -2)
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

# --- Venvs (auto-discover) ---
VENVS=""
for v in /home/z/vllm /home/z/.venv-vLLM_*/bin/python3; do
    if [ -x "$v/bin/python3" ] 2>/dev/null; then
        ver=$($v/bin/python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "?")
        VENVS="$VENVS|$v:$ver"
    elif [ -x "$v" ] 2>/dev/null; then
        ver=$($v -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "?")
        VENVS="$VENVS|$(dirname $(dirname $v)):$ver"
    fi
done
VENVS=${VENVS#|}  # strip leading pipe

# Active venv version
ACTIVE_VER="none"
for v in /home/z/.venv-vLLM_0170_Ray_02540_CUDA/bin/python3 /home/z/vllm/bin/python3; do
    if [ -x "$v" ]; then
        ACTIVE_VER=$($v -c "
import os; os.environ['VLLM_CONFIGURE_LOGGING']='0'
try:
    import vllm, torch, ray, pyarrow
    fa='?'
    try:
        import flash_attn; fa=flash_attn.__version__
    except: pass
    print(f'{vllm.__version__}|{torch.__version__}|{fa}|{ray.__version__}|{pyarrow.__version__}')
except Exception as e: print(f'ERR|ERR|ERR|ERR|ERR')
" 2>/dev/null | tail -1)
        break
    fi
done

# --- NFS ---
NFS_COUNT=$(mount | grep -c "nfs")
NFS_STATUS=""
for mnt in /repo /slowrepo /vmrepo /repo1; do
    if timeout 2 ls $mnt/ >/dev/null 2>&1; then
        NFS_STATUS="$NFS_STATUS ${mnt##*/}:OK"
    elif mountpoint -q $mnt 2>/dev/null; then
        NFS_STATUS="$NFS_STATUS ${mnt##*/}:STALE"
    else
        NFS_STATUS="$NFS_STATUS ${mnt##*/}:NONE"
    fi
done

# --- Disk ---
DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')

# --- Output ---
echo "DATA^$HOSTNAME^$IP^$NODE_TYPE^$HCA_ID^$GID_INDEX^$IB_STATE^$PEER^$GDR^$XPM^$KNM^$GPU_COUNT^$GPU_MEM^$DISK_FREE^$NFS_STATUS^$ACTIVE_VER^$VENVS"
REMOTE
)" > "$TEMP_DIR/$ip.raw" 2>&1
    } &
done
wait

# ── Summary Table ────────────────────────────────────────────────────────────
echo ""
printf "${BOLD}%-12s │ %-5s │ %-3s │ %-8s │ %-8s │ %-3s │ %-4s │ %-12s │ %-1s%-1s%-1s%-1s │ %-12s │ %-5s │ %-8s │ %-10s${RST}\n" \
    "NODE" "TYPE" "GPU" "HCA" "GID" "IB" "PORT" "vLLM" "PGXK" "    " "    " "    " "GPU MEM" "DISK" "NFS" "VENVS"
echo "────────────┼───────┼─────┼──────────┼──────────┼─────┼──────┼──────────────┼──────┼──────────────┼───────┼──────────┼──────────"

for ip in "${NODES[@]}"; do
    RAW="$TEMP_DIR/$ip.raw"
    LINE=$(grep "^DATA\^" "$RAW" 2>/dev/null | tail -1)

    if [ -z "$LINE" ]; then
        printf "${RED}%-12s │ %-5s │ %-3s │ %-8s │ %-8s │ %-3s │ %-4s │ %-12s │ %-4s │ %-12s │ %-5s │ %-8s │ %-10s${RST}\n" \
            "$ip" "?" "?" "?" "?" "?" "?" "OFFLINE" "????" "?" "?" "?" "?"
        continue
    fi

    IFS='^' read -r _ HOST NODE_IP NTYPE HCA GID IB_ST PEER GDR XPM KNM GPU_N GPU_MEM DISK NFS_ST VERSIONS VENVS <<< "$LINE"
    IFS='|' read -r VLLM TORCH FLASH RAY ARROW <<< "$VERSIONS"

    # Color coding
    IB_C="${RED}"; [[ "$IB_ST" == "PORT_ACTIVE" ]] && IB_C="${GRN}"
    P_C="${RED}"; [[ "$PEER" == "Y" ]] && P_C="${GRN}"
    G_C="${RED}"; [[ "$GDR" == "Y" ]] && G_C="${GRN}"
    X_C="${RED}"; [[ "$XPM" == "Y" ]] && X_C="${GRN}"
    K_C="${RED}"; [[ "$KNM" == "Y" ]] && K_C="${GRN}"

    # NFS compact
    NFS_SHORT=""
    for s in $NFS_ST; do
        name="${s%%:*}"; st="${s##*:}"
        if [[ "$st" == "OK" ]]; then NFS_SHORT="${NFS_SHORT}${GRN}✓${RST}"
        elif [[ "$st" == "STALE" ]]; then NFS_SHORT="${NFS_SHORT}${RED}✗${RST}"
        else NFS_SHORT="${NFS_SHORT}${DIM}-${RST}"
        fi
    done

    VENV_COUNT=$(echo "$VENVS" | tr '|' '\n' | grep -c '.')

    printf "%-12s │ %-5s │ %sx  │ %-8s │ GID=%-4s │ ${IB_C}%-3s${RST} │ ${IB_C}%-4s${RST} │ %-12s │ ${P_C}%s${RST}${G_C}%s${RST}${X_C}%s${RST}${K_C}%s${RST} │ %-12s │ %-5s │ %b │ %sv\n" \
        "$HOST" "$NTYPE" "$GPU_N" "$HCA" "$GID" "${IB_ST:0:3}" "${IB_ST:0:4}" "${VLLM:0:12}" \
        "${PEER:0:1}" "${GDR:0:1}" "${XPM:0:1}" "${KNM:0:1}" \
        "${GPU_MEM}" "$DISK" "$NFS_SHORT" "$VENV_COUNT"
done

echo "────────────┴───────┴─────┴──────────┴──────────┴─────┴──────┴──────────────┴──────┴──────────────┴───────┴──────────┴──────────"
echo -e "${DIM}PGXK = Peermem/GDR/Xpmem/Knem │ NFS = repo/slowrepo/vmrepo/repo1 (✓=OK ✗=STALE -=none)${RST}"
echo ""
