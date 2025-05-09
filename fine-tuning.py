import os
os.environ['WANDB_DISABLED'] = 'true'

from unsloth import FastLanguageModel
import torch

import json
from datasets import Dataset
from unsloth import to_sharegpt, standardize_sharegpt, apply_chat_template
def PrepareDataset(pathToFile):
    with open(pathToFile, encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset.from_dict({
        "instruction": [entry['instruction'] for entry in data],
        "output": [entry['output'] for entry in data],
    })

    dataset = to_sharegpt(
        dataset,
        merged_prompt = "{instruction}",
        output_column_name = "output",
        conversation_extension = 3, # Select more to handle longer conversations
    )

    dataset = standardize_sharegpt(dataset) 

    chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

    ### Instruction:
    {INPUT}

    ### Response:
    {OUTPUT}"""

    dataset = apply_chat_template(
        dataset,
        tokenizer = tokenizer,
        chat_template = chat_template,
        # default_system_message = "You are a helpful assistant", << [OPTIONAL]
    )

    return dataset

if __name__ == '__main__':
    with open('key.json', encoding='utf-8') as f:
        token = json.load(f)['hf_key']
    
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    model_name = 'utter-project/EuroLLM-9B-Instruct'

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token = token,
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    path_to_training_set = 'datasets/instruction_dataset_14873.json'
    train_dataset = PrepareDataset(path_to_training_set)
    eval_dataset = PrepareDataset('datasets/validation_dataset.json')

    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported

    learning_rate = 3e-4
    epochs = 5

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 8,
            warmup_steps = 300,
            num_train_epochs = epochs,
            learning_rate = learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"checkpoints/{model_name}",
            save_strategy = "epoch",
            eval_strategy = 'epoch',
        ),
    )

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    try:
        trainer_stats = trainer.train(resume_from_checkpoint = True)
    except:
        trainer_stats = trainer.train()

    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained_gguf(f"model/{model_name}-LatLeg-14.8K", tokenizer, quantization_method = ['f16'])
