import copy
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    CLIPImageProcessor,
    GemmaTokenizer
)

from data import LlavaDataset, TrainLLavaModelCollator, MyLlavaProcessor
from util import print_trainable_parameters
from peft import LoraConfig, get_peft_model

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=r"llava_model\model001")

@dataclass
class DataArguments:
    data_path: str = field(
        default=r"D:\Llava\data", metadata={"help": "Path to the training data."}
    )

def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = MyLlavaProcessor(image_processor=CLIPImageProcessor.from_pretrained(modelargs.model_name_or_path), tokenizer=GemmaTokenizer.from_pretrained(modelargs.model_name_or_path))
    
    LORA_R = 32
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["multi_modal_projector"],
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    return model, processor

def load_dataset(dataargs: DataArguments):
    llava_dataset = LlavaDataset(
        dataargs.data_path  # https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
    )
    return llava_dataset

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    data_collator = TrainLLavaModelCollator(processor, -100)
    train_dataset = load_dataset(data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()