import requests


class ServerClient:
    def __init__(self, host="localhost", port=5000):
        self.server_url = f"http://{host}:{port}"

    def start_llama_cpp_server(self, quantize: str):
        response = requests.post(f"{self.server_url}/start_llama_cpp_server", json={"quantize": quantize})
        if response.status_code == 200:
            print("Llama.cpp server started successfully")
        else:
            print(f"Failed to start Llama.cpp server: {response.text}")

    def kill_llama_cpp_server(self):
        response = requests.post(f"{self.server_url}/kill_llama_cpp_server")
        if response.status_code == 200:
            print("Llama.cpp server terminated successfully")
        else:
            print(f"Failed to terminate Llama.cpp server: {response.text}")

    def delete_gguf(self, quantize: str):
        response = requests.post(f"{self.server_url}/delete_gguf", json={"quantize": quantize})
        if response.status_code == 200:
            print(f"GGUF file for quantize {quantize} deleted successfully")
        else:
            print(f"Failed to delete GGUF file: {response.text}")

    def measure_usage_vram(self) -> float:
        response = requests.get(f"{self.server_url}/measure_usage_vram")
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                return data["vram_usage"]
            else:
                print(f"Failed to measure VRAM usage: {data['message']}")
                return -1
        else:
            print(f"Failed to measure VRAM usage: {response.text}")
            return -1
