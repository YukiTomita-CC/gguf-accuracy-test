import os
import subprocess

from flask import Flask, jsonify, request
import psutil


app = Flask(__name__)

@app.route('/start_llama_cpp_server', methods=['POST'])
def start_llama_cpp_server():
    quantize = request.json['quantize']
    cmd = f"./llama.cpp/build/bin/llama-server -m ./llama.cpp/models/Llama-3-Swallow-8B-Instruct-v0.1.{quantize}.gguf -c 1024 -n 1024 -ngl 33 --host 0.0.0.0 --port 8085"
    subprocess.Popen(cmd, shell=True)
    return jsonify({"status": "success", "message": "Server started"})

@app.route('/kill_llama_cpp_server', methods=['POST'])
def kill_llama_cpp_server():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'llama-server' in proc.info['name'] and '8085' in proc.info['cmdline']:
            proc.terminate()
            return jsonify({"status": "success", "message": "Server terminated"})
    return jsonify({"status": "error", "message": "Server not found"})

@app.route('/delete_gguf', methods=['POST'])
def delete_gguf():
    quantize = request.json['quantize']
    file_path = f"./llama.cpp/models/Llama-3-Swallow-8B-Instruct-v0.1.{quantize}.gguf"
    try:
        os.remove(file_path)
        return jsonify({"status": "success", "message": "File deleted"})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found"})

@app.route('/measure_usage_vram', methods=['GET'])
def measure_usage_vram():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        vram_usage = float(result.stdout.strip())
        return jsonify({"status": "success", "vram_usage": vram_usage})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
