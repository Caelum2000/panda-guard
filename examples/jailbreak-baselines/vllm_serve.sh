#!/bin/bash

trap "echo '收到中断信号，正在退出...'; exit" SIGINT

DEVICE_ID=7
PORT=8848
MODEL_NAME="Qwen/Qwen3-8B"
DTYPE="auto"
API_KEY="thb"

while true; do
    echo "启动 vLLM 服务：$MODEL_NAME"
    CUDA_VISIBLE_DEVICES=$DEVICE_ID vllm serve $MODEL_NAME --dtype $DTYPE --port $PORT --api-key $API_KEY

    echo "vLLM 服务已退出，5 秒后尝试重启..."
    sleep 5
done
