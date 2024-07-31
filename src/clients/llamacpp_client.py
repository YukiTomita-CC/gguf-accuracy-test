import json

from datasets import load_dataset
import requests
from tqdm import tqdm
from tqdm.contrib import tenumerate


class LlamacppClient:
    def __init__(self, host="localhost", port=8080):
        self.url = f"http://{host}:{port}"

    def generate_response(self, message:str, include_tps:bool=False):
        response = requests.post(
            f"{self.url}/completion",
            headers={
                'Content-Type': 'application/json'
            },
            data=json.dumps({
                'prompt': self._convert_llama3_prompt(message),
                'temperature': 0.6,
                'top_p': 0.9
            })
        )

        response_dict = json.loads(response.text)

        content = response_dict['content']
        tokens_per_second = round(response_dict['timings']['predicted_per_second'], 2)

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

            #WARN: OpenAIClientが参照するpathと一致させる必要があるがどちらにもハードコーディングになっている
            with open(f'data/model_responses/{quantize}/responses_{n+1}.json', 'w', encoding='utf-8') as f:
                json.dump({"responses": responses}, f, ensure_ascii=False, indent=2)
        
        return tps_list

    def _convert_llama3_prompt(self, message:str) -> str:
        prompt = f"""<|start_header_id|>system<|end_header_id|>

あなたは誠実で優秀な日本人のアシスタントです。userから特別指示がない限り日本語で回答してください。<|eot_id|><|start_header_id|>user<|end_header_id|>

{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
