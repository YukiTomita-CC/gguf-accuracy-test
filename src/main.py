import os
from time import sleep

from dotenv import load_dotenv

from clients.llamacpp_client import LlamacppClient
from clients.openai_client import OpenAIClient
from clients.server_client import ServerClient
from clients.spreadsheet_client import GoogleSSClient
from utils.batch_file_formatter import format_batch


def main():
    load_dotenv()
    
    host_address = os.environ.get('GGUF_TEST_SERVER')

    server_client = ServerClient(host=host_address)
    llamacpp_client = LlamacppClient(host=host_address)
    openai_client = OpenAIClient()
    google_client = GoogleSSClient()

    quantize_methods = ['Q8_0', 'Q6_K', ...]
    for q in quantize_methods:
        os.makedirs(f'data/eval_results/{q}', exist_ok=True)
        os.makedirs(f'data/model_responses/{q}', exist_ok=True)

        server_client.start_llama_cpp_server(q)
        sleep(15)
        
        tps_list = llamacpp_client.generate_responses_repeatedly(q, repeat_num=5)

        usage_vram = server_client.measure_usage_vram()
        server_client.kill_llama_cpp_server()


        batch_id = openai_client.create_batch(q)
        while True:
            if openai_client.check_batch_finish(batch_id):
                break

            sleep(300)

        batch_path = openai_client.download_batch(batch_id)

        data = format_batch(q, batch_path)

        google_client.write_to_spreadsheet(
            q,
            data["scores"],
            data["non_ja_responses"],
            data["infinite_repetitions"],
            tps_list,
            usage_vram)

        server_client.delete_gguf(q)


if __name__ == "__main__":
    main()
