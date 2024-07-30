import json
import statistics


def _process_jsonl(input_file, output_prefix):
    base_dict = {
        'reason': "",
        'grade': 1,
        'is_non_ja_response': False,
        'is_infinite_repetition': False
    }

    results = {}
    file_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            content = data['response']['body']['choices'][0]['message']['content']

            try:
                content_dict = json.loads(content)
            except json.JSONDecodeError:
                content_dict = base_dict.copy()

            try:
                results[str(i % 100)] = {
                    "reason": content_dict['reason'],
                    "grade": int(content_dict['grade']),
                    "is_non_ja_response": content_dict['is_non_ja_response'],
                    "is_infinite_repetition": content_dict['is_infinite_repetition']
                }
            except KeyError:
                results[str(i % 100)] = base_dict.copy()

            if (i + 1) % 100 == 0 or i == 499:
                file_count += 1
                output_file = f"{output_prefix}_{file_count}.json"
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(results, out_f, ensure_ascii=False, indent=2)
                results = {}

def format_batch(quantize, file):
    result_dir = f'data/eval_results/{quantize}/results'

    _process_jsonl(file, result_dir)

    scores = []
    non_ja_responses_list = []
    infinite_repetitions_list = []
    for m in range(5):
        with open(f'{result_dir}_{m+1}.json', 'r', encoding='utf-8') as f:
            results = json.load(f)

        grade = []
        non_ja_responses = 0
        infinite_repetitions = 0
        for i in range(100):
            grade.append(results[str(i)]["grade"])
            if results[str(i)]["is_non_ja_response"]:
                non_ja_responses += 1
            if results[str(i)]["is_infinite_repetition"]:
                infinite_repetitions += 1
        
        scores.append(statistics.mean(grade))

        non_ja_responses_list.append(non_ja_responses)
        infinite_repetitions_list.append(infinite_repetitions)

    return {
        "scores": scores,
        "non_ja_responses": statistics.mean(non_ja_responses_list),
        "infinite_repetitions": statistics.mean(infinite_repetitions_list)
    }
