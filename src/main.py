import argparse
import os
from time import sleep

from dotenv import load_dotenv

from clients.llamacpp_client import LlamacppClient
# from clients.openai_client import OpenAIClient
from clients.server_client import ServerClient
from clients.spreadsheet_client import GoogleSSClient
# from utils.batch_file_formatter import format_batch


def main(quantize:str, port1:int, port2:int):
    load_dotenv()

    server_client = ServerClient(port=port1)
    llamacpp_client = LlamacppClient(port=port2)
    # openai_client = OpenAIClient()
    # google_client = GoogleSSClient()

    quantize_methods = [quantize]

    # quantize_methods = [
    #     "Q8_0", "Q6_K", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    #     "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    #     "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q2_K"
    # ]

    # quantize_methods = [
    #     "Q6_K", "Q5_K_S", "Q5_K_M", "Q4_K_S", "Q4_K_M", "IQ4_XS", "IQ4_NL",
    #     "Q3_K_S", "Q3_K_M", "Q3_K_L", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    #     "Q2_K_S", "Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M"
    # ]

    for q in quantize_methods:
        os.makedirs(f'data/eval_results/{q}', exist_ok=True)
        os.makedirs(f'data/model_responses/{q}', exist_ok=True)

        server_client.start_llama_cpp_server(q)
        sleep(600)
        
        tps_list = llamacpp_client.generate_responses_repeatedly(q, repeat_num=5)

        usage_vram = server_client.measure_usage_vram()
        server_client.kill_llama_cpp_server()

        # result_file_path = openai_client.create_batch(q)

        # data = format_batch(q, result_file_path)

        print(tps_list)
        print(usage_vram)

        # google_client.write_to_spreadsheet(
        #     q,
        #     data["scores"],
        #     data["non_ja_responses"],
        #     data["infinite_repetitions"],
        #     tps_list,
        #     usage_vram,
        #     prefix="")

        server_client.delete_gguf(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the app with specified port and quantization size.')

    parser.add_argument('--port1', type=int, default=5000, help='Port to run the flask server on (default: 5000)')
    parser.add_argument('--port2', type=int, default=8080, help='Port to run the llamacpp server on (default: 8080)')
    parser.add_argument('quantization_size', type=str, help='Quantization size for the model (e.g., Q8_0, Q4_K_M)')

    args = parser.parse_args()

    main(port1=args.port1, port2=args.port2, quantize=args.quantization_size)
