import json
import time
from tqdm import tqdm
from tqdm.contrib import tenumerate
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from datasets import load_dataset


class ResponsesCreater:
    def __init__(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained("cyberagent/Llama-3.1-70B-Japanese-Instruct-2407", device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("cyberagent/Llama-3.1-70B-Japanese-Instruct-2407")

        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate_response(self, message:str, include_tps:bool=False):
        messages = [
            {"role": "user", "content": message}
        ]

        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        start_time = time.time()
        output_ids = self.model.generate(input_ids,
            max_new_tokens=1024,
            streamer=self.streamer
            )
        end_time = time.time()

        execution_time = end_time - start_time
        
        content = self.tokenizer.decode(output_ids[0][len(input_ids[0]):])

        tokens_per_second = (len(output_ids[0]) - len(input_ids[0])) / execution_time

        if include_tps:
            return (content, tokens_per_second)
        else:
            return (content, 0)

    def generate_responses_repeatedly(self, quantize:str, path_or_HFrepo:str="data/test_data", repeat_num:int=5):
        dataset = load_dataset(path_or_HFrepo)
        subset = dataset["test"]
        dataset_rows = len(subset)

        tps_list = []
        for n in tqdm(range(repeat_num)):
            responses = []
            for i, example in tenumerate(subset):
                input = example["input"]

                if (n+1) == repeat_num and (dataset_rows-5) <= i <= (dataset_rows-1):
                    output, tps = self.generate_response(input, include_tps=True)
                    tps_list.append(tps)
                else:
                    output = self.generate_response(input)[0]

                responses.append(
                    {
                        "id": str(i+1),
                        "input": input,
                        "output": output
                    })

            with open(f'data/model_responses/{quantize}/responses_{n+1}.json', 'w', encoding='utf-8') as f:
                json.dump({"responses": responses}, f, ensure_ascii=False, indent=2)
        
        return tps_list
