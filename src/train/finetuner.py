import gc
import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer


class Finetuner:

    def __init__(self, model_id: str, dataset_id: str, model_class, dataset_name: str = None,
                 finetuned_modelname: str = "opt-125-exit-code", max_context_length: int = 256,
                 use_steps: bool = True, save_steps: int = 500, epochs: int = 3,
                 gradient_accumulation_steps: int = 4, learning_rate: float = 1e-5,
                 weight_decay: float = 0.5e-5,
                 batch_size: int = 8, output_dir: str = "cached/opt-1.3B-exit-code", debug: bool = False,
                 exit_indices: list[int] = None):

        self.model_id = model_id
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.model_class = model_class

        self.finetuned_modelname = finetuned_modelname
        self.max_context_length = max_context_length
        self.use_steps = use_steps
        self.save_steps = save_steps
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.DEBUG = debug
        self.exit_indices = exit_indices
        self.weight_decay = weight_decay
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._initialize_components()
        self._setup_logging()

    def _initialize_components(self):
        self.model = self.model_class.from_pretrained(self.model_id, mode="finetune", exit_indices=self.exit_indices)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, add_prefix_space=True)
        self.dataset = self._load_dataset()

        if self.device == 'cuda' and self.DEBUG:
            print("GPU/cuda available ")
            torch.cuda.empty_cache()
            print("GPU Memory cleaned")
        elif self.DEBUG:
            print("No GPU available switching to CPU")
        gc.collect()

    def _load_dataset(self):
        if self.dataset_name:
            return load_dataset(self.dataset_id, self.dataset_name)
        return load_dataset(self.dataset_id)

    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler('training.log')
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        self.logger = logger

    def _remove_special_tokens(self, example):
        example['code'] = [token for token in example['code'] if token not in ["<s>", "</s>"]]
        return example

    def _tokenize_function(self, batch):
        tokenized_examples = []
        for example in batch["code"]:
            if example:
                tokenized_example = self.tokenizer(
                    example,
                    is_split_into_words=True,
                    truncation=False,
                )["input_ids"]
                tokenized_examples.append(tokenized_example)

        if not tokenized_examples:
            return {"input_ids": []}

        eos_token_id = self.tokenizer.eos_token_id
        concatenated_examples = []
        for tokenized_example in tokenized_examples:
            concatenated_examples.extend(tokenized_example)

        if concatenated_examples and concatenated_examples[-1] == eos_token_id:
            concatenated_examples = concatenated_examples[:-1]

        input_batch = [
            concatenated_examples[i:i + self.max_context_length]
            for i in range(0, len(concatenated_examples), self.max_context_length)
            if len(concatenated_examples[i:i + self.max_context_length]) == self.max_context_length
        ]

        if self.DEBUG:
            print(f"Total tokens: {len(concatenated_examples)}")
            print(f"Number of chunks: {len(input_batch)}")

        return {"input_ids": input_batch}

    def _prepare_dataset(self):
        dataset = self.dataset.map(self._remove_special_tokens)
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        if self.DEBUG:
            print(tokenized_dataset)
            for i in range(25):
                print(tokenized_dataset["train"][i])
                print(self.tokenizer.decode(tokenized_dataset["train"][i]["input_ids"]))
        return tokenized_dataset

    def _get_training_arguments(self):
        if self.use_steps:
            print("here")
            return TrainingArguments(
                eval_strategy="steps",
                save_strategy="steps",
                output_dir=self.output_dir,
                logging_dir=self.output_dir,
                logging_strategy="steps",
                save_steps=self.save_steps,
                eval_steps=self.save_steps,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                learning_rate=self.learning_rate,
                lr_scheduler_type='constant',
                num_train_epochs=self.epochs,
                weight_decay=self.weight_decay,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
            )
        else:
            print("---batchsize:---")
            print(self.batch_size)
            print("----")
            return TrainingArguments(
                eval_strategy="epoch",
                save_strategy="epoch",
                output_dir=self.output_dir,
                logging_dir=self.output_dir,
                logging_strategy="epoch",
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                learning_rate=self.learning_rate,
                lr_scheduler_type='constant',
                num_train_epochs=self.epochs,
                weight_decay=self.weight_decay,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
            )

    def finetune(self):
        tokenized_dataset = self._prepare_dataset()
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        training_args = self._get_training_arguments()

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"]
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.logger.info(f"Model saved at {self.output_dir}")
        print(f"Model saved at {self.output_dir}")
        return self.output_dir
