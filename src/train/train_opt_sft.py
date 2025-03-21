from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from src.models.opt.modeling_opt_single_head_fixed_exit import OPTEESingleHead

set_seed(42)

model_name = "facebook/opt-2.7b"
dataset_name = "google/code_x_glue_cc_code_completion_token"
use_fast = True if "opt" in model_name else False
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=use_fast,
    add_prefix_space=True
)
tokenizer.pad_token = tokenizer.eos_token

epochs = 5
batch_size = 8
bf16 = False
fp16 = True
gradient_accumulation_steps = 16
context_length = 256
learning_rate = 1e-5
weight_decay = 0.01

out_dir = 'XXX'


def dataset_array_to_string(example):
    example["code"] = ' '.join(example["code"][1:-1])
    return example


dataset = load_dataset(dataset_name, 'java')
dataset = dataset.map(dataset_array_to_string)

model = OPTEESingleHead.from_pretrained(model_name, exit_indices=[3, 5, 7, 9, 11, 13, 15, 19, 23, 27],
                                        exits_first_half=[3, 5, 7, 9, 11, 13, 15],
                                        exits_second_half=[19, 23, 27], total_weight_first_half=0.7,
                                        total_weight_second_half=0.2,
                                        mode="finetune", weight_mode="geometric", log_loss=False,
                                        cache_dir="X")

training_args = TrainingArguments(
    output_dir=f"{out_dir}/logs",
    eval_strategy='epoch',
    logging_steps=5,
    weight_decay=weight_decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_strategy='steps',
    save_strategy='epoch',
    bf16=bf16,
    fp16=fp16,
    report_to='tensorboard',
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    lr_scheduler_type='constant',
    num_train_epochs=epochs,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    dataset_text_field='code',
    max_seq_length=context_length,
    tokenizer=tokenizer,
    args=training_args,

    packing=True
)

trainer.train()
