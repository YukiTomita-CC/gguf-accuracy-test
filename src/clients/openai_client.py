import json
from time import sleep

from datasets import load_dataset
from openai import OpenAI


class OpenAIClient:
    def __init__(self) -> None:
        self.openai_client = OpenAI()
        self.batch = None

    def create_batch(self, quantize:str):
        #TODO: LlamacppClientのgenerate_responses_repeatedlyと同様に選べるように
        dataset = load_dataset("data/test_data")
        test_set = dataset["test"]

        evaluation_prompt = """問題, 正解例, 採点基準, 言語モデルが生成した回答が与えられます。

    「採点基準」と「正解例」を参考にして、「言語モデルの回答」を評価してください。
    そして、回答理由および1,2,3,4,5の5段階評価による採点結果、以下の2つのチェックを、「評価フォーマット」に示すようなJSON形式で返してください。
    - チェック1: 問題の指示にはないにもかかわらず、すべて日本語ではない言語で答えている(英語での補足や部分的に日本語ではない場合は構わない) -> "is_non_ja_response"をtrueとする
    - チェック2: 同じ文字や文字列が連続して止まらなくなっている(例:「ああああああ」、「十, 十, 十\n- う: う, う」等) -> "is_infinite_repetition"をtrueとする

    # 問題
    {input_text}

    # 正解例
    {output_text}

    # 採点基準
    基本的な採点基準
    - 1点: 誤っている、 指示に従えていない
    - 2点: 誤っているが、方向性は合っている
    - 3点: 部分的に誤っている、 部分的に合っている
    - 4点: 合っている
    - 5点: 役に立つ

    基本的な減点項目
    - 不自然な日本語: -1点
    - 部分的に事実と異なる内容を述べている: -1点

    問題固有の採点基準
    {eval_aspect}

    # 言語モデルの回答
    {pred}

    # 評価フォーマット
    {{
        "reason": "(採点基準に照らした評価内容)",
        "grade": (採点結果、1～5の5段階評価),
        "is_non_ja_response": "(デフォルトはfalse、上記のチェック1に合致すればtrue)",
        "is_infinite_repetition": "(デフォルトはfalse、上記のチェック2に合致すればtrue)"
    }}
    """

        tasks = []
        for n in range(5):
            with open(f"data/model_responses/{quantize}/responses_{n+1}.json", 'r', encoding='utf-8') as f:
                j = json.load(f)
                model_response = j["responses"]

            for i, example in enumerate(test_set):
                content = evaluation_prompt.format(
                    input_text=example["input"],
                    output_text=example["output"],
                    eval_aspect=example["eval_aspect"],
                    pred=model_response[i]["output"]
                )
                
                task = {
                    "custom_id": f"task-{n}-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "response_format": { 
                            "type": "json_object"
                        },
                        "messages": [
                            {"role": "system", "content": "あなたは言語モデルの採点者です。"},
                            {"role": "user", "content": content}
                        ],
                    }
                }
                
                tasks.append(task)

        output_file_name = f"data/batches/upload/{quantize}.jsonl"
        with open(output_file_name, 'w', encoding='utf-8') as file:
            for obj in tasks:
                file.write(json.dumps(obj, ensure_ascii=False) + '\n')

        batch_file = self.openai_client.files.create(
            file=open(output_file_name, "rb"),
            purpose="batch"
            )

        self.batch = self.openai_client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
            )
        
        while True:
            print("check")
            if self._check_batch_finish():
                break

            sleep(30)
        
        result_file_path = self._download_batch(quantize)

        return result_file_path
    
    def _check_batch_finish(self):
        self.batch = self.openai_client.batches.retrieve(self.batch.id)

        if self.batch.status == 'completed':
            return True

        return False

    def _download_batch(self, quantize:str):
        result_file_id = self.batch.output_file_id
        result = self.openai_client.files.content(result_file_id).content

        result_file_name = f"data/batches/download/{quantize}.jsonl"

        with open(result_file_name, 'wb') as file:
            file.write(result)
        
        return result_file_name
