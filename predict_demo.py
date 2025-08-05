import os
from typing import List
import pdb
import json
import sys
from tqdm import tqdm
import argparse
import logging
from swift.llm import VllmEngine, PtEngine
from swift.llm import InferEngine, InferRequest, RequestConfig
from swift.plugin import InferStats


# 固定写死 官网才能看到相关日志
log_file_path = '/home/admin/workspace/job/logs/user.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

result = []

def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    logging.info(f"dataset split succes, now infering....")
    request_config = RequestConfig(max_tokens=1820, temperature=0.6, repetition_penalty=1.1)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])

    for index, response in enumerate(resp_list):
        dict = {}
        res = resp_list[index].choices[0].message.content
        logging.info(f"llm response: {res}")
        dict['text'] = res
        result.append(dict)

if __name__ == '__main__':
    try:
        logging.info(f"success in to predic scrip, now loading user model......")
        parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")
        parser.add_argument("--model_path", type=str, default="/home/admin/predict/user-model-v3")
        parser.add_argument("--data_path", type=str, default="/")
        parser.add_argument("--output_path", type=str, default="/")
        parser.add_argument("--model_type", type=str, default="qwen2_5")
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        args = parser.parse_args()

        model_path = args.model_path
        model_type = args.model_type
        output_path = args.output_path
        tensor_parallel_size = args.tensor_parallel_size

        model = model_path

        infer_backend = 'vllm'
        logging.info(f"param model path: {model_path}")
        logging.info(f"param outputpath: {output_path}")
        if infer_backend == 'pt':
            engine = PtEngine(model, model_type=model_type, max_batch_size=64)
        elif infer_backend == 'vllm':
            engine = VllmEngine(model, model_type=model_type, gpu_memory_utilization=0.8, tensor_parallel_size=tensor_parallel_size)

        logging.info(f"user model load success now begin split dataset")
        dataset = []

        with open(args.data_path,'r',encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))

        res = []
        for idx, data in tqdm(enumerate(dataset)):
            input = data['conversations'][0]['value']

            data_new = {}
            data_new['messages'] = []
            dict = {}
            dict['role'] = 'user'
            dict['content'] = input

            data_new['messages'].append(dict)
            res.append(InferRequest(**data_new))

        infer_requests = res
        infer_batch(engine, infer_requests)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in result:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
        logging.info(f"infer success, file saved path: {output_path}")
    except Exception as e:
        logging.error(f"error: {e}")
