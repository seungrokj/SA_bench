import json
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, choices=['h100', 'h200', 'b200'], required=True)
parser.add_argument('-p', '--print-results', action='store_true', default=False)
args = parser.parse_args()

import os
# Notice, Change model directory, START
if args.gpu == 'h100':
    MODEL_DIR= '/scratch/huggingface_models'
    assert os.path.exists(MODEL_DIR)
elif args.gpu == 'h200':
    MODEL_DIR= '/scratch1/models'
    assert os.path.exists(MODEL_DIR)
elif args.gpu == 'b200':
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
        return

    network_name = 'bmk-net'
    server_name = 'bmk-server'
    port = 8100
    image_name = 'vllm/vllm-openai:v0.9.1'
    image_name_client = 'rocm/pytorch-private:vllm-openai_v0.9.1_client'

    dist_backend = '--distributed-executor-backend mp' if tp_size > 1 else ''
    #quant_flags = '--dtype bfloat16 --quantization fbgemm_fp8' if 'FP8' in model_name else ''
    quant_flags = ''
    max_model_len = int((input_len + output_len) * 1.125)

    script = f'''
docker network create {network_name}

docker run --rm -d --network {network_name} --name {server_name} \
    --runtime nvidia --gpus all --ipc host --privileged --ulimit memlock=-1 --ulimit stack=67108864 \
    -v {MODEL_DIR}:/model \
    -v "$PWD/.hf_cache/":/root/.cache/huggingface/hub/ -v "$PWD/.vllm_cache/":/root/.cache/vllm/ -e HF_TOKEN="$(cat hf_token.txt)" \
    {image_name} \
        --model {model_name} --port {port} \
        --tensor-parallel-size {tp_size} --distributed-executor-backend mp \
        --max-num-seqs {max_num_seqs} --gpu-memory-utilization 0.95 \
        --max-model-len {max_model_len} --max-seq-len-to-capture {max_model_len} \
        --disable-log-requests

printf 'RESULT_FILENAME: %s\n' "{result_filename}"
while ! docker logs {server_name} 2>&1 | grep -q "Application startup complete."; do
    sleep 1
done

docker run --rm -t --network {network_name} --name bmk-client \
    --runtime nvidia \
    -v {MODEL_DIR}:/model \
    -v $PWD:/workspace/ -w /workspace/vllm/benchmarks/ -e HF_TOKEN="$(cat hf_token.txt)" \
    --entrypoint "/usr/bin/python3" {image_name_client} \
    benchmark_serving.py \
        --model {model_name} --backend vllm --base-url "http://{server_name}:{port}" \
        --dataset-name "random" --random-input-len {input_len} --random-output-len {output_len} --random-prefix-len 0 \
        --num-prompts $(( {max_concurrency} * 10 )) --max-concurrency {max_concurrency} --request-rate "inf" --ignore-eos \
        --save-result --result-dir "/workspace/results/" --result-filename "{result_filename}.json" --percentile-metrics "ttft,tpot,itl,e2el"

docker stop {server_name}; docker network rm {network_name}
'''
    subprocess.run(script, shell=True, check=True)


def launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency, max_num_seqs):
    result_filename = f'dsv3_tp{tp_size}_isl{input_len}_osl{output_len}_c{max_concurrency}_s{max_num_seqs}'
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
    image_name = 'lmsysorg/sglang:v0.4.6.post4-cu124'
    network_name = 'bmk-net'
    server_name = 'bmk-server'
    port = 8100

    script = f'''#!/usr/bin/env bash
docker network create {network_name}

docker run --rm -d --network {network_name} --name {server_name} \
    --runtime nvidia --gpus all --ipc host --privileged --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$PWD/.hf_cache/":/root/.cache/huggingface/hub/ -v "$PWD/.inductor_cache/":/tmp/torchinductor_root/ -e HF_TOKEN="$(cat hf_token.txt)" \
    -v "$PWD/.dg_cache/":/root/.cache/deep_gemm/ -e SGL_ENABLE_JIT_DEEPGEMM=1 \
    {image_name} \
    python3 -m sglang.launch_server --model-path {model_name} --host 0.0.0.0 --port {port} --tp {tp_size} --trust-remote-code

printf 'RESULT_FILENAME: %s\n' "{result_filename}"
while ! docker logs {server_name} 2>&1 | grep -q "The server is fired up and ready to roll!"; do
    sleep 1
done

docker run --rm -t --network {network_name} --name bmk-client \
    --runtime nvidia \
    -v $PWD:/workspace/ -w /workspace/vllm/benchmarks/ -e HF_TOKEN="$(cat hf_token.txt)" \
    --entrypoint "/usr/bin/python3" {image_name_client} \
    benchmark_serving.py \
        --model {model_name} --backend sglang --base-url "http://{server_name}:{port}" \
        --dataset-name "random" --random-input-len {input_len} --random-output-len {output_len} --random-prefix-len 0 \
        --num-prompts $(( {max_concurrency} * 10 )) --max-concurrency {max_concurrency} --request-rate "inf" --ignore-eos \
        --save-result --result-dir "/workspace/results/" --result-filename "{result_filename}.json" --percentile-metrics "ttft,tpot,itl,e2el"

docker rename {server_name} {server_name}-old; docker stop {server_name}-old; docker network rm {network_name}
'''
    subprocess.run(script, shell=True, check=True)


if args.gpu == 'h100':
    max_num_batched_tokens = 8192  # V1 engine default
    for input_len, output_len in [(1024, 1024), (1024, 4096), (4096, 1024)]:
        t_s = time.time()

        if 1:
            # LLaMA 70B
            #for tp_size in [4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Meta-Llama-3.1-70B-Instruct', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # LLaMA 405B FP8
            tp_size = 8
            for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                launch_bmk_llama('/model/Llama-3.1-405B-FP8', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)

        t_e = time.time()
        print(f'ISL{input_len}/OSL{output_len} BENCHMARK TIME ELAPSED: {((t_e - t_s) / 60.0):.2f} minutes')

elif args.gpu == 'h200':
    max_num_batched_tokens = 8192  # V1 engine default

    for input_len, output_len in [(1024, 1024), (1024, 4096), (4096, 1024)]:
        t_s = time.time()

        if 1:
            # LLaMA 70B
            #for tp_size in [2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # LLaMA 405B FP8
            #for tp_size in [4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Meta-Llama-3.1-405B-FP8/', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)

        if 0:
            # DeepseekV3
            tp_size = 8
            for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency, max_concurrency)

        t_e = time.time()
        print(f'ISL{input_len}/OSL{output_len} BENCHMARK TIME ELAPSED: {((t_e - t_s) / 60.0):.2f} minutes')

elif args.gpu == 'b200':
    max_num_batched_tokens = 8192  # V1 engine default

    for input_len, output_len in [(1024, 1024), (1024, 4096), (4096, 1024)]:
        t_s = time.time()

        if 1:
            # LLaMA 70B
            #for tp_size in [2, 4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Llama-3.1-70B-Instruct', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)
        if 0:
            # LLaMA 405B FP8
            #for tp_size in [4, 8]:
            for tp_size in [8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_llama('/model/Meta-Llama-3.1-405B-FP8/', input_len, output_len, tp_size, max_concurrency, max_concurrency, max_num_batched_tokens)

        if 0:
            # DeepseekV3
            tp_size = 8
            for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency, max_concurrency)

        t_e = time.time()
        print(f'ISL{input_len}/OSL{output_len} BENCHMARK TIME ELAPSED: {((t_e - t_s) / 60.0):.2f} minutes')

else:
    raise ValueError(f'Unknown GPU {args.gpu}')
