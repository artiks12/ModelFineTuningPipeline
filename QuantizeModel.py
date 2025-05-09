import subprocess

def QuantizeModel(path, modelName, methods):
    fullPath = path + '/' + modelName

    executable = './/llama.cpp/build/bin/Release/llama-quantize.exe'
    source_path = f'{fullPath}/unsloth.F16.gguf'

    for method in methods:
        gguf_filename = f'unsloth.{method.upper()}.gguf'
        target_path = f'{fullPath}/{gguf_filename}'

        with open(f'{fullPath}/Modelfile','r') as f:
            modelfile = f.read()

        with open(f'{fullPath}/Modelfile_{method.upper()}','w') as f:
            f.write(modelfile.replace('unsloth.F16.gguf',gguf_filename))

        command = [executable, source_path, target_path, method, '40']

        subprocess.run(command)

path = f'path/to/model/folder'
modelName = 'modelName'
methods = ['q4_k_m']

QuantizeModel(path,modelName,methods)
