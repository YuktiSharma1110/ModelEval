import pandas as pd
from rouge import Rouge
import numpy as np
import difflib


def calculate_lcs(x, y):
    # Create a sequence matcher
    seq = difflib.SequenceMatcher(None, x, y)
    # Find the matching blocks, extract the longest one
    match = max(seq.get_matching_blocks(), key=lambda block: block.size)
    # Return the longest matching substring
    return x[match.a: match.a + match.size]


def calculate_rouge_and_lcs(data):
    rouge = Rouge()
    results = []

    for index, row in data.iterrows():
        if pd.isna(row['generated_query']):
            results.append({
                'query': row['query'],
                'generated_query': '',
                'LCS': '',
                'rouge-l_r': None,
                'rouge-l_p': None,
                'rouge-l_f': None
            })
            continue

        # Compute the ROUGE scores
        score = rouge.get_scores(row['generated_query'], row['query'])[0]['rouge-l']

        # Calculate LCS
        lcs = calculate_lcs(row['query'], row['generated_query'])

        results.append({
            'query': row['query'],
            'generated_query': row['generated_query'],
            'LCS': lcs,
            'rouge-l_r': score['r'],
            'rouge-l_p': score['p'],
            'rouge-l_f': score['f']
        })

    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    file_path = 'gemma7b_finetuned_model_output-gemma7b_finetuned_model_output.csv'
    file_path_1 = 'gemma7b_full_output-gemma7b_full_output.csv'
    file_path_2 = 'swe_llama_outputs.csv'
    file_path_3 = 'gemma2b_finetuned_output-gemma2b_finetuned_output.csv'

    data = pd.read_csv(file_path_2)

    # Calculate ROUGE scores and LCS, then convert to DataFrame
    results_df = calculate_rouge_and_lcs(data)

    # Save the detailed ROUGE scores and LCS to a CSV file
    detailed_output_path = 'detailed_rouge_scores_with_lcs.csv'
    results_df.to_csv(detailed_output_path, index=False)
    print(f'Detailed ROUGE scores with LCS saved to {detailed_output_path}')

    # Calculate average ROUGE scores
    average_rouge_scores = results_df[['rouge-l_r', 'rouge-l_p', 'rouge-l_f']].mean().to_dict()
    print("Average ROUGE Scores for Gemma 7B Fine Tuned Dataset:")
    for key, value in average_rouge_scores.items():
        print(f"{key}: {value:.3f}")

    # Save the average scores to a CSV file
    average_scores_df = pd.DataFrame([average_rouge_scores])
    average_output_path = 'average_rouge_scores.csv'
    average_scores_df.to_csv(average_output_path, index=False)
    print(f'Average ROUGE scores saved to {average_output_path}')
