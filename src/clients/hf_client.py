import json
import time

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from tqdm import tqdm
from tqdm.contrib import tenumerate


class HFClient:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("cyberagent/calm3-22b-chat", device_map="auto", torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm3-22b-chat")

    def generate_response(self, message:str, include_tps:bool=False):
        messages = [
            {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
            {"role": "user", "content": message}
        ]

        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        start = time.time()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.5
            )
        end = time.time()
        
        output = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

        print(f"Input: {message}")
        print(f"Output: {output}")
        print(f"all tokens: {len(output_ids[0])}")
        print(f"input tokens: {len(input_ids[0])}")
        print(f"output tokens: {len(output_ids[0][input_ids.shape[-1]:])}")
        print("Check 'all tokens = input tokens + output tokens'")


        content = output
        tokens_per_second = round(len(output_ids[0][input_ids.shape[-1]:]) / (end - start), 2)

        if include_tps:
            return (content, tokens_per_second)
        else:
            return (content, 0)

    def generate_responses_repeatedly(self, quantize:str, path_or_HFrepo:str="data/test_data", repeat_num:int=5):
        dataset = load_dataset(path_or_HFrepo)
        # TODO: データによってはtrainの場合もあるので選べるようにした方がいい？
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
                
                break

            #WARN: OpenAIClientが参照するpathと一致させる必要があるがどちらにもハードコーディングになっている
            with open(f'data/model_responses/{quantize}/responses_{n+1}.json', 'w', encoding='utf-8') as f:
                json.dump({"responses": responses}, f, ensure_ascii=False, indent=2)
            
            break
        
        return tps_list


if __name__ == "__main__":
    client = HFClient()
    client.generate_responses_repeatedly("original")
