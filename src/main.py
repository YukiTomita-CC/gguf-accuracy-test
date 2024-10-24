import os
from time import sleep

from dotenv import load_dotenv

from clients.llamacpp_client import LlamacppClient
from clients.openai_client import OpenAIClient
from clients.server_client import ServerClient
# from clients.spreadsheet_client import GoogleSSClient
from utils.batch_file_formatter import format_batch


def main():
    load_dotenv()
    
    # host_address = os.environ.get('GGUF_TEST_SERVER')

    server_client = ServerClient()
    llamacpp_client = LlamacppClient()
    openai_client = OpenAIClient()
    # google_client = GoogleSSClient()

    quantize_methods = [
        "Q4_K_M", "Q5_K_M", "Q6_K", "bf16"
    ]
    for q in quantize_methods:
        os.makedirs(f'data/eval_results/{q}', exist_ok=True)
        os.makedirs(f'data/model_responses/{q}', exist_ok=True)

        server_client.start_llama_cpp_server(q)
        sleep(300)
        
        tps_list = llamacpp_client.generate_responses_repeatedly(q, repeat_num=5)

        usage_vram = server_client.measure_usage_vram()
        server_client.kill_llama_cpp_server()

        result_file_path = openai_client.create_batch(q)

        data = format_batch(q, result_file_path)

        print("=== Start of Results ===")
        print("Quantize method:", q)
        print("Scores:", data["scores"])
        print("TPS:", tps_list)
        print("Usage VRAM:", usage_vram)
        print("=== End of Results ===")

        # google_client.write_to_spreadsheet(
        #     q,
        #     data["scores"],
        #     data["non_ja_responses"],
        #     data["infinite_repetitions"],
        #     tps_list,
        #     usage_vram,
        #     prefix="")

        # server_client.delete_gguf(q)


if __name__ == "__main__":
    main()
