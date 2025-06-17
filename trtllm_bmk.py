import json
import subprocess
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
    MODEL_DIR='/mnt/nvme2n1/models'
    assert os.path.exists(MODEL_DIR)
# Notice, END

if args.print_results:
    print(f'config, tp, conc, mnbt, ttft, tpot, itl, e2el, total_tput')


def launch_bmk_trt(model_name, input_len, output_len, tp_size, max_concurrency, max_num_tokens):
    #model_handle = model_name.split('/')[1].split('-')[2].lower()
    #model_handle = model_name.split('/')[1].split('-')[2].lower()
    model_handle = model_name.split('/')[2].split('-')[2].lower()

    result_filename = f'trt_{model_handle}_tp{tp_size}_isl{input_len}_osl{output_len}_c{max_concurrency}_mnt{max_num_tokens}'
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
        return

    network_name = 'bmk-net'
    server_name = 'bmk-server3'
    port = 8100
    image_name = 'nvcr.io/nvidia/tensorrt-llm/release:0.21.0rc1'
    image_name_client = 'rocm/pytorch-private:vllm-openai_v0.9.1_client'

    script = f'''#!/usr/bin/env bash
docker network create {network_name}

docker run --rm -d --network {network_name} --name {server_name} \
    --runtime nvidia --gpus all --ipc host --privileged --ulimit memlock=-1 --ulimit stack=67108864 \
    -v {MODEL_DIR}:/model \
    -v "$PWD/.hf_cache/":/root/.cache/huggingface/hub/ -v "$PWD/configs_trtllm/":/root/config/ -e HF_TOKEN="$(cat hf_token.txt)" \
    {image_name} \
    trtllm-serve serve {model_name} --host 0.0.0.0 --port {port} --backend pytorch --tp_size {tp_size} --max_num_tokens {max_num_tokens} \
    --extra_llm_api_options /root/config/extra_llm_options.yml

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


if args.gpu == 'b200':
    #for input_len, output_len, max_num_tokens in [(1024, 1024, 2500), (1024, 4096, 5500), (4096, 1024, 5500)]:
    for input_len, output_len, max_num_tokens in [(1024, 1024, 2500)]:
        if 0:
            for tp_size in [2, 4, 8]:
                for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
                    launch_bmk_trt('meta-llama/Llama-3.1-70B', input_len, output_len, tp_size, max_concurrency, max_num_tokens)
        #for tp_size in [4, 8]:
        for tp_size in [8]:
            #for max_concurrency in [4, 8, 16, 32, 64, 128, 256]:
            for max_concurrency in [16, 32, 64, 128, 256]:
                launch_bmk_trt('/model/Llama-3.1-405B-Instruct-FP4', input_len, output_len, tp_size, max_concurrency, max_num_tokens)
