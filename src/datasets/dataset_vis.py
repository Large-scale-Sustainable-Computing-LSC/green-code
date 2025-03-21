import re
from collections import Counter

import datasets
import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from datasets import concatenate_datasets, load_dataset
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('punkt')

dataset_train = load_dataset("google/code_x_glue_cc_code_completion_token", name="python")['train']
dataset_test = load_dataset("google/code_x_glue_cc_code_completion_token", name="python")['test']

combined_dataset = concatenate_datasets([dataset_train, dataset_test])
dataset = combined_dataset
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})


def dataset_array_to_string(example):
    example["code"] = ' '.join(example["code"][1:-1])
    return example


dataset = dataset.map(dataset_array_to_string)


def tokenize_code(text):
    return word_tokenize(text)


# Token length distribution
def plot_token_length_distribution(dataset):
    token_lengths = [len(tokenize_code(item['code'])) for item in dataset]

    plt.figure(figsize=(10, 6))
    sns.histplot(token_lengths, bins=150, kde=True)
    plt.title('Distribution of Code Token Lengths')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()


def plot_wordcloud(dataset):
    all_text = " ".join([item['code'] for item in dataset])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Code Snippets")
    plt.show()


def plot_top_words(dataset, top_n=20):
    all_text = " ".join([item['code'] for item in dataset])
    words = tokenize_code(all_text)
    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)

    labels, counts = zip(*most_common)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(labels), palette='Blues_d')
    plt.title(f'Top {top_n} Most Frequent Tokens')
    plt.xlabel('Count')
    plt.ylabel('Token')
    plt.show()


# Average token length per sample
def plot_avg_token_length_per_sample(dataset):
    avg_token_lengths = [np.mean([len(token) for token in tokenize_code(item['code'])]) for item in dataset]

    plt.figure(figsize=(10, 6))
    sns.histplot(avg_token_lengths, bins=50, kde=True)
    plt.title('Distribution of Average Token Lengths per Code Snippet')
    plt.xlabel('Average Token Length')
    plt.ylabel('Frequency')
    plt.show()


def plot_special_character_frequency(dataset, special_chars=[';', '{', '}', '(', ')']):
    all_text = " ".join([item['code'] for item in dataset])
    special_char_counts = {char: all_text.count(char) for char in special_chars}

    labels, counts = zip(*special_char_counts.items())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(labels), palette='Reds_d')
    plt.title('Frequency of Special Characters')
    plt.xlabel('Count')
    plt.ylabel('Character')
    plt.show()


def plot_mean_import_package_token_distribution(dataset, language="java"):
    percentages_per_sample = []
    absolute_import_package_tokens = []
    if language == "java":
        for item in dataset:
            code = item['code']

            segments = code.replace('}', ';').split(';')

            total_tokens = 0
            import_package_tokens = 0

            for segment in segments:
                segment = segment.strip()

                segment_tokens = segment.split()
                total_tokens += len(segment_tokens)

                if segment.startswith('import') or segment.startswith('package'):
                    import_package_tokens += len(segment_tokens)

            if total_tokens > 0:
                import_package_percentage = (import_package_tokens / total_tokens) * 100
            else:
                import_package_percentage = 0

            percentages_per_sample.append(import_package_percentage)
            absolute_import_package_tokens.append(import_package_tokens)
    elif language == "python":

        for item in dataset:
            code = item['code']

            segments = code.split('<EOL>')

            total_tokens = 0
            import_package_tokens = 0

            for segment in segments:
                segment = segment.strip()

                segment_tokens = segment.split()
                total_tokens += len(segment_tokens)

                if segment.startswith('import') or segment.startswith('from'):
                    import_package_tokens += len(segment_tokens)

            if total_tokens > 0:
                import_package_percentage = (import_package_tokens / total_tokens) * 100
            else:
                import_package_percentage = 0

            percentages_per_sample.append(import_package_percentage)
            absolute_import_package_tokens.append(import_package_tokens)

    mean_percentage = np.mean(percentages_per_sample)
    std_deviation = np.std(percentages_per_sample)
    mean_absolute_tokens = np.mean(absolute_import_package_tokens)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie([mean_percentage, 100 - mean_percentage],
           labels=[f'Import/Package Tokens',
                   f'Other Tokens'],
           autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])

    plt.subplots_adjust(bottom=0.2)

    plt.show()


def plot_token_type_distribution(dataset):
    keyword_pattern = r'\b(if|else|for|while|return|public|private|protected|class)\b'
    operator_pattern = r'[+\-*/=<>!&|^%~]'
    literal_pattern = r'\b\d+\b'

    keyword_count = 0
    operator_count = 0
    literal_count = 0
    other_count = 0

    for item in dataset:
        code = item['code']
        keyword_count += len(re.findall(keyword_pattern, code))
        operator_count += len(re.findall(operator_pattern, code))
        literal_count += len(re.findall(literal_pattern, code))
        other_count += len(tokenize_code(code)) - (keyword_count + operator_count + literal_count)

    labels = ['Keywords', 'Operators', 'Literals', 'Others']
    counts = [keyword_count, operator_count, literal_count, other_count]

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Token Type Distribution')
    plt.show()


def calculate_cyclomatic_complexity(code):
    decision_points = len(re.findall(r'\b(if|for|while|case|catch|switch)\b', code))
    return decision_points + 1


def plot_code_complexity(dataset, max_complexity=2000):
    complexities = [calculate_cyclomatic_complexity(item['code']) for item in dataset]

    plt.figure(figsize=(10, 6))
    sns.histplot(complexities, bins=20, kde=True)
    plt.title('Distribution of Cyclomatic Complexity')
    plt.xlabel('Cyclomatic Complexity')
    plt.ylabel('Frequency')
    plt.xlim(0, max_complexity)
    plt.show()


def calculate_nesting_depth(code):
    depth = 0
    max_depth = 0
    for line in code.splitlines():
        if re.search(r'\b(if|for|while|switch|try)\b', line):
            depth += 1
            max_depth = max(max_depth, depth)
        elif '}' in line:
            depth -= 1
    return max_depth


def plot_nesting_depth_distribution(dataset):
    depths = [calculate_nesting_depth(item['code']) for item in dataset]

    plt.figure(figsize=(10, 6))
    sns.histplot(depths, bins=20, kde=True)
    plt.title('Distribution of Maximum Nesting Depth')
    plt.xlabel('Nesting Depth')
    plt.ylabel('Frequency')
    plt.show()


def plot_token_length_distribution_limited(dataset, max_length=5000):
    token_lengths = [len(item['code'].split()) for item in dataset if len(item['code'].split()) <= max_length]

    plt.figure(figsize=(10, 6))
    sns.histplot(token_lengths, bins=50, kde=True)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()


def min_max_length(dataset):
    max_len = 0
    min_len = 1000000
    for item in dataset:
        code = item['code']
        tokens = tokenize_code(code)
        max_len = max(max_len, len(tokens))
        min_len = min(min_len, len(tokens))

    print(f"Max Length: {max_len}")
    print(f"Min Length: {min_len}")


def percentage_of_length_examples(dataset, max_len):
    token_lengths = [len(item['code'].split()) for item in dataset]
    num_examples = len(token_lengths)
    examples_lower_length = [length for length in token_lengths if length > max_len]
    num_examples_lower_length = len(examples_lower_length)
    percentage = (num_examples_lower_length / num_examples) * 100
    print(f"Percentage of examples with more than {max_len} tokens: {percentage:.2f}%")


# Visualize the dataset with different figures
def visualize_code_dataset(dataset):
    #  plot_mean_import_package_token_distribution(dataset)
    #  print(min_max_length(dataset))
    #  percentage_of_length_examples(dataset, 5000)
    print("Visualizing Code Token Length Distribution...")
    # plot_token_length_distribution(dataset)

    print("\nGenerating Word Cloud...")
    #  plot_wordcloud(dataset)

    print("\nVisualizing Top Words...")
    #   plot_top_words(dataset)

    print("\nVisualizing Average Token Length per Sample...")
    #  plot_avg_token_length_per_sample(dataset)

    print("\nVisualizing Special Character Frequency...")
    # plot_special_character_frequency(dataset)

    plot_token_length_distribution_limited(dataset, max_length=5000)
    # plot_token_type_distribution(dataset)
    # plot_code_complexity(dataset)
    # plot_nesting_depth_distribution(dataset)
    plot_mean_import_package_token_distribution(dataset, language="python")


visualize_code_dataset(dataset)
