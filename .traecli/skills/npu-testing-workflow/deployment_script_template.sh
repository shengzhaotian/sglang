#!/bin/bash
#
# NPU Model Service Deployment Script Template
# This script provides a complete template for deploying SGLang models on Ascend NPU devices.
# Customize the variables below according to your specific requirements.
#

set -e

#############################################
# Section 1: Configuration Variables
#############################################

MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
MODEL_NAME=$(basename "$MODEL_PATH")

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROUTER_PORT="${ROUTER_PORT:-6688}"

DEPLOY_MODE="${DEPLOY_MODE:-mixed}"

TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"

MEM_FRACTION="${MEM_FRACTION:-0.8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

QUANTIZATION="${QUANTIZATION:-modelslim}"
DTYPE="${DTYPE:-bfloat16}"

CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-"8 16 24 32"}"

PREFILL_IPS=("${PREFILL_IPS[@]:-}")
DECODE_IPS=("${DECODE_IPS[@]:-}")

DISAGGREGATION_BOOTSTRAP_PORT="${DISAGGREGATION_BOOTSTRAP_PORT:-8995}"
MF_STORE_PORT="${MF_STORE_PORT:-24667}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

#############################################
# Section 2: System Performance Settings
#############################################

configure_system_performance() {
    echo "Configuring system performance settings..."
    
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true
    sudo sysctl -w vm.swappiness=0 || true
    sudo sysctl -w kernel.numa_balancing=0 || true
    sudo sysctl -w kernel.sched_migration_cost_ns=50000 || true
}

#############################################
# Section 3: CANN Environment Setup
#############################################

setup_cann_environment() {
    echo "Setting up CANN environment..."
    
    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    else
        echo "Warning: CANN toolkit environment script not found"
    fi
    
    if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh
    else
        echo "Warning: NNAL ATB environment script not found"
    fi
    
    if [ -d "/usr/local/Ascend/8.5.0/compiler/bishengir/bin" ]; then
        export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH
    fi
    
    if [ -f "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash" ]; then
        source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
    fi
}

#############################################
# Section 4: Environment Variables
#############################################

configure_environment_variables() {
    echo "Configuring environment variables..."
    
    export SGLANG_SET_CPU_AFFINITY=1
    
    unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY 2>/dev/null || true
    unset ASCEND_LAUNCH_BLOCKING 2>/dev/null || true
    
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export STREAMS_PER_DEVICE=32
    
    export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1600}"
    export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
}

configure_npu_optimizations() {
    echo "Configuring NPU-specific optimizations..."
    
    export SGLANG_NPU_USE_MLAPO="${SGLANG_NPU_USE_MLAPO:-1}"
    export SGLANG_USE_FIA_NZ="${SGLANG_USE_FIA_NZ:-1}"
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}"
    export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
}

configure_deepep() {
    echo "Configuring DeepEP for MoE models..."
    
    export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-32}"
    export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}"
    export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-16}"
    export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
}

#############################################
# Section 5: Server Launch Functions
#############################################

launch_pd_mixed_server() {
    echo "Launching PD Mixed mode server..."
    
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --tp-size "$TP_SIZE" \
        --dp-size "$DP_SIZE" \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --quantization "$QUANTIZATION" \
        --dtype "$DTYPE" \
        --mem-fraction-static "$MEM_FRACTION" \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --cuda-graph-bs $CUDA_GRAPH_BS \
        --watchdog-timeout 9000 \
        --disable-radix-cache \
        --chunked-prefill-size -1 \
        --moe-a2a-backend deepep \
        --deepep-mode auto \
        --enable-dp-attention \
        --enable-dp-lm-head \
        ${SPECULATIVE_ARGS:-} \
        2>&1 | tee "$LOG_DIR/server_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
}

launch_prefill_server() {
    local node_rank=$1
    local prefill_ip="${PREFILL_IPS[$node_rank]}"
    
    echo "Launching Prefill server on $prefill_ip (rank $node_rank)..."
    
    export ASCEND_MF_STORE_URL="tcp://${PREFILL_IPS[0]}:${MF_STORE_PORT}"
    export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1536}"
    export TASK_QUEUE_ENABLE=2
    
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$prefill_ip" \
        --port "$PORT" \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port $((DISAGGREGATION_BOOTSTRAP_PORT + node_rank)) \
        --disaggregation-transfer-backend ascend \
        --tp-size "$TP_SIZE" \
        --dp-size 2 \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --quantization "$QUANTIZATION" \
        --dtype "$DTYPE" \
        --mem-fraction-static "${MEM_FRACTION:-0.6}" \
        --max-running-requests 8 \
        --context-length 8192 \
        --disable-radix-cache \
        --chunked-prefill-size -1 \
        --max-prefill-tokens 28680 \
        --moe-a2a-backend deepep \
        --deepep-mode normal \
        --enable-dp-attention \
        --disable-shared-experts-fusion \
        ${SPECULATIVE_ARGS:-} \
        2>&1 | tee "$LOG_DIR/prefill_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
}

launch_decode_server() {
    local node_rank=$1
    local decode_ip="${DECODE_IPS[$node_rank]}"
    
    echo "Launching Decode server on $decode_ip (rank $node_rank)..."
    
    export ASCEND_MF_STORE_URL="tcp://${PREFILL_IPS[0]}:${MF_STORE_PORT}"
    export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-720}"
    export TASK_QUEUE_ENABLE=1
    export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
    
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$decode_ip" \
        --port "$PORT" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend ascend \
        --tp-size $((TP_SIZE * 2)) \
        --dp-size $((DP_SIZE * 2)) \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --quantization "$QUANTIZATION" \
        --dtype "$DTYPE" \
        --mem-fraction-static "${MEM_FRACTION:-0.8}" \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --cuda-graph-bs $CUDA_GRAPH_BS \
        --moe-a2a-backend deepep \
        --deepep-mode low_latency \
        --enable-dp-attention \
        --enable-dp-lm-head \
        --moe-dense-tp 1 \
        --watchdog-timeout 9000 \
        --context-length 8192 \
        --prefill-round-robin-balance \
        --disable-shared-experts-fusion \
        --tokenizer-worker-num 4 \
        --load-balance-method decode_round_robin \
        ${SPECULATIVE_ARGS:-} \
        2>&1 | tee "$LOG_DIR/decode_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
}

launch_router() {
    echo "Launching SGLang Router..."
    
    local prefill_args=""
    for i in "${!PREFILL_IPS[@]}"; do
        prefill_args="$prefill_args --prefill http://${PREFILL_IPS[$i]}:$PORT $((DISAGGREGATION_BOOTSTRAP_PORT + i))"
    done
    
    local decode_args=""
    for decode_ip in "${DECODE_IPS[@]}"; do
        decode_args="$decode_args --decode http://${decode_ip}:$PORT"
    done
    
    export SGLANG_DP_ROUND_ROBIN=1
    
    python -m sglang_router.launch_router \
        --pd-disaggregation \
        --policy cache_aware \
        $prefill_args \
        $decode_args \
        --host "$HOST" \
        --port "$ROUTER_PORT" \
        --mini-lb \
        2>&1 | tee "$LOG_DIR/router_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
}

#############################################
# Section 6: Main Execution
#############################################

detect_local_role() {
    local local_host1=$(hostname -I | awk '{print $1}')
    local local_host2=$(hostname -I | awk '{print $2}')
    
    for i in "${!PREFILL_IPS[@]}"; do
        if [[ "$local_host1" == "${PREFILL_IPS[$i]}" || "$local_host2" == "${PREFILL_IPS[$i]}" ]]; then
            echo "prefill:$i"
            return
        fi
    done
    
    for i in "${!DECODE_IPS[@]}"; do
        if [[ "$local_host1" == "${DECODE_IPS[$i]}" || "$local_host2" == "${DECODE_IPS[$i]}" ]]; then
            echo "decode:$i"
            return
        fi
    done
    
    echo "unknown"
}

main() {
    echo "=========================================="
    echo "NPU Model Service Deployment"
    echo "Model: $MODEL_NAME"
    echo "Mode: $DEPLOY_MODE"
    echo "=========================================="
    
    configure_system_performance
    setup_cann_environment
    configure_environment_variables
    configure_npu_optimizations
    configure_deepep
    
    case "$DEPLOY_MODE" in
        "mixed")
            launch_pd_mixed_server
            ;;
        "prefill")
            local role=$(detect_local_role)
            if [[ "$role" == prefill:* ]]; then
                local rank=${role#prefill:}
                launch_prefill_server "$rank"
            else
                echo "Error: This node is not configured as a prefill node"
                exit 1
            fi
            ;;
        "decode")
            local role=$(detect_local_role)
            if [[ "$role" == decode:* ]]; then
                local rank=${role#decode:}
                launch_decode_server "$rank"
            else
                echo "Error: This node is not configured as a decode node"
                exit 1
            fi
            ;;
        "router")
            launch_router
            ;;
        *)
            echo "Error: Unknown deployment mode: $DEPLOY_MODE"
            echo "Valid modes: mixed, prefill, decode, router"
            exit 1
            ;;
    esac
}

main "$@"
