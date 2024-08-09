import argparse

from huggingface_hub import HfApi, login


def main(quantize:str):
    login(token="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", add_to_git_credential=True)

    api = HfApi()

    for i in range(1, 6):
        api.upload_file(
            path_or_fileobj=f"./data/model_responses/{quantize}/responses_{i}.json",
            path_in_repo=f"non-quantize/model_responses/{quantize}/responses_{i}.json",
            repo_id="YukiTomita-CC/temp_data",
            repo_type="dataset",
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the app on a specified port.')
    parser.add_argument('--quantize', type=str, help='quantize')
    args = parser.parse_args()

    main(quantize=args.quantize)
