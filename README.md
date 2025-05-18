# ModelFineTuningPipeline
This is the repository to fine-tune models. It is a part of the master thesis "Evaluation and Adaptation of Large Language Models for Question-Answering on Legislation" made in University of Latvia.

### How to Use
This script was used with Python 3.10 so it is recomended to use this version of python. You also need to do these things:
- Install the unsloth package.
- Download llama.cpp code and binaries: https://github.com/ggml-org/llama.cpp
- Put training and validation data in datasets folder. Repository that creates these datasets is available here: https://github.com/artiks12/DatasetPreperation
- Specify your HuggingFace key in key.json file

Since there is a bug in unsloth code that prevents you from quantizing models in any format other than F16, a separate file for model quantization is created (QuantizeModel.py). You need to specify the path to your model GGUF file and quantization methods and run the script.
