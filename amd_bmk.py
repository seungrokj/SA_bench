import json
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, choices=['mi300x', 'mi325x', 'mi355x'], required=True)
parser.add_argument('-p', '--print-results', action='store_true', default=False)
args = parser.parse_args()

import os
# Notice, Change model directory, START
if args.gpu == 'mi300x':
    MODEL_DIR="/home/amd/models"
    assert os.path.exists(MODEL_DIR)
elif args.gpu == 'mi325x':
    MODEL_DIR=""
    assert os.path.exists(MODEL_DIR)
elif args.gpu == 'mi355x':
    MODEL_DIR=""
    assert os.path.exists(MODEL_DIR)
# Notice, END

if args.print_results:
    print(f'config, tp, conc, mnbt, ttft, tpot, itl, e2el, total_tput')


def launch_bmk_llama(model_name, input_len, output_len, tp_size, max_concurrency, max_num_seqs, max_num_batched_tokens):
    model_handle = '70b' if '70' in model_name else '405b'
    result_filename = (
            f'{model_handle}_tp{tp_size}_isl{input_len}_osl{output_len}_'
            f'c{max_concurrency}_s{max_num_seqs}_mnbt{max_num_batched_tokens}'
    )
    result_file_path = Path(f'results/{result_filename}.json')

    if args.print_results:
        if not result_file_path.exists():
            return
        fields = ['median_ttft_ms', 'median_tpot_ms', 'median_itl_ms', 'median_e2el_ms', 'total_token_throughput']
        with open(result_file_path) as f:
            results = json.load(f)
        print(f'{result_filename}, {tp_size}, {max_concurrency}, {max_num_batched_tokens},', ', '.join(f'{results[f]:.3f}' for f in fields))
        return

    if result_file_path.exists():
        print(f'Skipping {result_filename}')
        return

    network_name = 'bmk-net'
    server_name = 'bmk-server'
    port = 8100
    image_name = 'rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605'

    dist_backend = '--distributed-executor-backend mp' if tp_size > 1 else ''
    quant_flags = '--dtype bfloat16 --quantization fp8' if 'FP8' in model_name else ''
    max_model_len = int((input_len + output_len) * 1.125)
    full_graph = '\'{"full_cuda_graph": true}\''


    script = f'''#!/usr/bin/env bash
docker network create {network_name}

docker run --rm -d --network {network_name} --ipc host --name {server_name} \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e NCCL_MIN_NCHANNELS=112 \
    -e VLLM_USE_V1=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
    -v {MODEL_DIR}:/model \
    -v "$PWD/.hf_cache/":/root/.cache/huggingface/hub/ -v "$PWD/.vllm_cache/":/root/.cache/vllm/ -e HF_TOKEN="$(cat hf_token.txt)" \
    {image_name} \
    vllm serve {model_name} --port {port} \
        --tensor-parallel-size {tp_size} {dist_backend} {quant_flags} \
        --max-num-batched-tokens {max_num_batched_tokens} \
        --max-num-seqs {max_num_seqs} --gpu-memory-utilization 0.95 \
        --max-model-len {max_model_len} --max-seq-len-to-capture {max_model_len} \
        --disable-log-requests \
        --compilation-config {full_graph}

printf "RESULT_FILENAME=%s\n" "{result_filename}"
while ! docker logs {server_name} 2>&1 | grep -q "Application startup complete."; do
    sleep 1
    if docker logs {server_name} 2>&1 | grep -q "ERROR"; then
        docker logs {server_name} >& "failed_runs/{result_filename}.log"
        docker stop {server_name}; docker network rm {network_name}
        exit 1
    fi
done

docker run --rm -t --network {network_name} --name bmk-client \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v {MODEL_DIR}:/model \
    -v $PWD:/workspace/ -w /workspace/vllm/benchmarks/ -e HF_TOKEN=$(cat hf_token.txt) \
    {image_name} \
        python benchmark_serving.py \
            --model {model_name} --backend vllm --base-url "http://{server_name}:{port}" \
            --dataset-name "random" --random-input-len {input_len} --random-output-len {output_len} --random-prefix-len 0 \
            --num-prompts $(( {max_concurrency} * 10 )) --max-concurrency {max_concurrency} --request-rate "inf" --ignore-eos \
            --save-result --result-dir "/workspace/results/" --result-filename "{result_filename}.json" --percentile-metrics "ttft,tpot,itl,e2el"

docker stop {server_name}; docker network rm {network_name}
sleep 60
'''
    subprocess.run(script, shell=True, check=True)


def launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency):
    result_filename = f'dsv3_tp{tp_size}_isl{input_len}_osl{output_len}_c{max_concurrency}'
    result_file_path = Path(f'results/{result_filename}.json')

    if args.print_results:
        if not result_file_path.exists():
            return
        fields = ['median_ttft_ms', 'median_tpot_ms', 'median_itl_ms', 'median_e2el_ms', 'total_token_throughput']
        with open(result_file_path) as f:
            results = json.load(f)
        print(f'{result_filename}, {tp_size}, {max_concurrency}, -1,', ', '.join(f'{results[f]:.3f}' for f in fields))
        return

    if result_file_path.exists():
        print(f'Skipping {result_filename}')
        return

    model_name = 'deepseek-ai/DeepSeek-V3'
    network_name = 'bmk-net'
    server_name = 'bmk-server'
    port = 8100
    image_name = 'rocm/sgl-dev:upstream_20250422'

    script = f'''#!/usr/bin/env bash
docker network create {network_name}

docker run --rm -d --network {network_name} --ipc host --name {server_name} \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$PWD/.hf_cache/":/root/hf_cache/ -v "$PWD/.inductor_cache/":/tmp/torchinductor_root/ \
    -e HF_HUB_CACHE=/root/hf_cache/ -e HF_TOKEN="$(cat hf_token.txt)" -e SGLANG_AITER_MOE=1 \
    {image_name} \
    python3 -m sglang.launch_server --model-path {model_name} --host 0.0.0.0 --port {port} --tp {tp_size} --trust-remote-code \
    --chunked-prefill-size 131072 --enable-torch-compile --torch-compile-max-bs 256

printf "RESULT_FILENAME=%s\n" "{result_filename}"
while ! docker logs {server_name} 2>&1 | grep -q "The server is fired up and ready to roll!"; do
    sleep 1
done

docker run --rm -t --network {network_name} --name bmk-client \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v $PWD:/workspace/ -w /workspace/vllm/benchmarks/ -e HF_TOKEN=$(cat hf_token.txt) \
    rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250410 \
        python benchmark_serving.py \
            --model {model_name} --backend vllm --base-url "http://{server_name}:{port}" \
            --dataset-name "random" --random-input-len {input_len} --random-output-len {output_len} --random-prefix-len 0 \
            --num-prompts $(( {max_concurrency} * 10 )) --max-concurrency {max_concurrency} --request-rate "inf" --ignore-eos \
            --save-result --result-dir "/workspace/results/" --result-filename "{result_filename}.json" --percentile-metrics "ttft,tpot,itl,e2el"

docker stop {server_name}; docker network rm {network_name}
sleep 60
'''
    subprocess.run(script, shell=True, check=True)


if args.gpu == 'mi300x':
    max_num_batched_tokens = 8192  # V1 engine default

    for input_len, output_len in [(1024, 1024), (1024, 4096), (4096, 1024)]:
        t_s = time.time()
        if 0:
            # LLaMA 70B FP8
            #for tp_size in [1, 2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 1:
            # LLaMA 70B
            #for tp_size in [1, 2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 1:
            # LLaMA 405B FP8
            #for tp_size in [4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-405B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # DeepseekV3
            tp_size = 8
            for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency)
        t_e = time.time()
        print(f'ISL{input_len}/OSL{output_len} BENCHMARK TIME ELAPSED: {((t_e - t_s) / 60.0):.2f} minutes')
elif args.gpu == 'mi325x':
    max_num_batched_tokens = 8192  # V1 engine default

    for input_len, output_len in [(1024, 1024), (1024, 4096), (4096, 1024)]:
        t_s = time.time()
        if 0:
            # LLaMA 70B FP8
            #for tp_size in [1, 2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 1:
            # LLaMA 70B
            #for tp_size in [1, 2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # LLaMA 405B FP8
            #for tp_size in [4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-405B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # DeepseekV3
            tp_size = 8
            for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency)
        t_e = time.time()
        print(f'ISL{input_len}/OSL{output_len} BENCHMARK TIME ELAPSED: {((t_e - t_s) / 60.0):.2f} minutes')
elif args.gpu == 'mi355x':
    max_num_batched_tokens = 8192  # V1 engine default

    for input_len, output_len in [(1024, 1024), (1024, 4096), (4096, 1024)]:
        t_s = time.time()
        if 0:
            # LLaMA 70B FP8
            #for tp_size in [1, 2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 1:
            # LLaMA 70B
            #for tp_size in [1, 2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # LLaMA 405B FP8
            #for tp_size in [4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-405B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # DeepseekV3
            tp_size = 8
            for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency)
        t_e = time.time()
        print(f'ISL{input_len}/OSL{output_len} BENCHMARK TIME ELAPSED: {((t_e - t_s) / 60.0):.2f} minutes')
else:
    raise ValueError(f'Unknown GPU {args.gpu}')
