import pandas as pd
import re
import ast

def tokenize_sql(sql):
    """Tokenize the SQL query to include all symbols and non-word characters as separate tokens."""
    if pd.isna(sql):
        return []
    tokens = re.findall(r'\w+|[^\w\s]', sql)  # \w+ for words, [^\w\s] for non-word characters excluding spaces
    return [token.lower() for token in tokens]

def compare_components(ref_tokens, gen_tokens):
    """Calculate the percentage of reference tokens present in the generated tokens and return True/False for each token."""
    gen_token_set = set(gen_tokens)  # Set of generated tokens (already lowercase)
    results = {}
    matched_tokens = 0
    for token in ref_tokens:
        presence = token in gen_token_set
        results[token] = presence
        if presence:
            matched_tokens += 1
    total_ref_tokens = len(ref_tokens)
    match_percentage = (matched_tokens / total_ref_tokens * 100) if total_ref_tokens > 0 else 0
    return match_percentage, results

def evaluate_sql(data):
    """Evaluate SQL queries by comparing tokens of reference queries to generated queries, handling tokenization."""
    data['generated_tokens'] = data['generated_query'].apply(tokenize_sql)
    data['reference_tokens'] = data['query_toks'].apply(
        lambda x: [token.lower() for token in (ast.literal_eval(x) if isinstance(x, str) else x)]
    )
    match_percentages = []
    match_details = []
    exact_matches = []
    for _, row in data.iterrows():
        if pd.isna(row['generated_query']):
            match_percentages.append(0)
            match_details.append({})
            exact_matches.append(0)
        else:
            percentage, details = compare_components(row['reference_tokens'], row['generated_tokens'])
            match_percentages.append(percentage)
            match_details.append(details)
            # Determine if the match is exact: all tokens must be True
            exact_matches.append(1 if all(details.values()) else 0)
    return match_percentages, match_details, exact_matches

if __name__ == "__main__":
    # Load the dataset
    file_path = 'gemma7b_finetuned_model_output-gemma7b_finetuned_model_output.csv'
    file_path_1 = 'gemma7b_full_output-gemma7b_full_output.csv'
    file_path_2 = 'swe_llama_outputs.csv'
    file_path_3 = 'gemma2b_finetuned_output-gemma2b_finetuned_output.csv'

    data = pd.read_csv(file_path_2)

    # Evaluate the SQL and append the results
    results = evaluate_sql(data)
    data['partial_match_percentage'], data['match_details'], data['exact_match'] = results

    # Calculate overall average of partial matching and percentage of exact matches labeled as 1
    average_partial_match = sum(data['partial_match_percentage']) / len(data['partial_match_percentage'])
    percentage_exact_match = (sum(data['exact_match']) / len(data['exact_match'])) * 100

    # Optionally, print results or save to a file
    print(f"Average Partial Match Percentage: {average_partial_match:.2f}%")
    print(f"Percentage of Exact Matches (Label=1): {percentage_exact_match:.2f}%")
   # print(data[['query', 'generated_query', 'partial_match_percentage', 'match_details', 'exact_match']])
    output_path = 'evaluation_results_detailed.csv'
    data.to_csv(output_path, index=False)
    print(f'Results saved to {output_path}')




