import re

import datasets
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

nltk.download('punkt')


class CodeCompletionEvaluator:
    def __init__(self, model_id, dataset, device="cuda", max_input_tokens=10):
        """
        Initializes the evaluator with the model, dataset, and device.

        Parameters:
            model_id (str): The path or Hugging Face model ID.
            dataset (datasets.Dataset): The dataset containing the code samples.
            device (str): The device to run the model on ('cuda' or 'cpu').
            max_input_tokens (int): The number of tokens from the method to use as input for code generation.
        """
        self.dataset = dataset
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.max_input_tokens = max_input_tokens

    def extract_methods_from_java(self, content):
        """
        Extracts all methods from a Java file, including their signatures and bodies.

        Parameters:
            content (str): The content of the Java file as a string.

        Returns:
            list: A list of strings, where each string is a method definition including its body.
        """
        # Regular expression to capture methods
        method_pattern = re.compile(r'''
            (public|protected|private|static|final|synchronized|abstract|\s)*          # method modifiers
            (\w+\s+)+                                                                # return type
            (\w+)                                                                     # method name
            \s*\(([^)]*)\)                                                            # parameters inside parentheses
            \s*(throws\s+[^\{]+)?                                                      # optional 'throws' clause
            \s*\{([^}]+)\}                                                           # method body inside curly braces
        ''', re.VERBOSE)

        methods = []
        for match in method_pattern.finditer(content):
            method = match.group(0)
            methods.append(method)
        return methods

    def generate_code_from_model(self, method_signature):
        """
        Generates code completions for the given method signature using the Hugging Face model.

        Parameters:
            method_signature (str): The method signature to provide as input for code completion.

        Returns:
            str: The generated method code from the model.
        """
        inputs = self.tokenizer(method_signature, return_tensors="pt", truncation=True,
                                max_length=self.max_input_tokens).to(self.device)

        outputs = self.model.generate(inputs["input_ids"], max_length=512, num_return_sequences=1,
                                      pad_token_id=self.tokenizer.eos_token_id)
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code

    def calculate_bleu(self, predicted_methods, reference_methods):
        """
        Calculate the BLEU score for the predicted methods compared to the reference methods.

        Parameters:
            predicted_methods (list): A list of strings representing the predicted methods.
            reference_methods (list): A list of strings representing the reference methods.

        Returns:
            float: The BLEU score.
        """
        reference_corpus = [[nltk.word_tokenize(method)] for method in reference_methods]
        predicted_corpus = [nltk.word_tokenize(method) for method in predicted_methods]
        return corpus_bleu(reference_corpus, predicted_corpus)

    def calculate_rouge(self, predicted_methods, reference_methods):
        """
        Calculate ROUGE scores for the predicted methods compared to the reference methods.

        Parameters:
            predicted_methods (list): A list of strings representing the predicted methods.
            reference_methods (list): A list of strings representing the reference methods.

        Returns:
            dict: A dictionary with ROUGE scores.
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for predicted, reference in zip(predicted_methods, reference_methods):
            score = scorer.score(reference, predicted)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)

        avg_scores = {key: sum(value) / len(value) for key, value in scores.items()}
        return avg_scores

    def evaluate(self, sample_index=0):
        """
        Evaluates the model on a specific sample from the dataset by extracting methods,
        generating code, and calculating BLEU and ROUGE scores.

        Parameters:
            sample_index (int): The index of the sample in the dataset to evaluate.

        Returns:
            dict: A dictionary containing the BLEU and ROUGE scores.
        """
        sample = self.dataset[sample_index]
        code_input = sample['code']

        reference_methods = self.extract_methods_from_java(code_input)
        print(reference_methods)

        predicted_methods = []
        for method in reference_methods:
            method_signature = method.split('{')[0].strip()  # Extract the method signature
            generated_code = self.generate_code_from_model(method_signature)
            predicted_methods.append(generated_code)

        bleu_score = self.calculate_bleu(predicted_methods, reference_methods)

        rouge_scores = self.calculate_rouge(predicted_methods, reference_methods)

        return {"bleu_score": bleu_score, "rouge_scores": rouge_scores}
