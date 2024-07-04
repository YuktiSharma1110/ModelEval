import pandas as pd
import re
import ast
import matplotlib.pyplot as plt

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

def categorize_query_by_length(query):
    """Categorize the complexity of SQL queries based on their length."""
    length = len(query.split())
    if length <= 10:
        return 'Short'
    elif length <= 25:
        return 'Medium'
    else:
        return 'Long'

if __name__ == "__main__":
    # Load the dataset
    file_path = 'gemma7b_finetuned_model_output-gemma7b_finetuned_model_output.csv'
    data = pd.read_csv(file_path)

    # Categorize queries by length
    data['length_category'] = data['query'].apply(categorize_query_by_length)

    # Save categorized queries to separate files
    for category in data['length_category'].unique():
        subset = data[data['length_category'] == category]
        subset.to_csv(f'queries_{category.lower()}.csv', index=False)

    # Initialize a dictionary to hold the results
    final_results = {}

    # Perform analysis on each category file
    categories = ['Short', 'Medium', 'Long']
    for category in categories:
        category_data = pd.read_csv(f'queries_{category.lower()}.csv')
        match_percentages, match_details, exact_matches = evaluate_sql(category_data)
        category_data['partial_match_percentage'], category_data['match_details'], category_data['exact_match'] = match_percentages, match_details, exact_matches

        # Calculate overall average of partial matching and percentage of exact matches labeled as 1
        average_partial_match = sum(category_data['partial_match_percentage']) / len(category_data['partial_match_percentage'])
        percentage_exact_match = (sum(category_data['exact_match']) / len(category_data['exact_match'])) * 100

        # Save results in the dictionary
        final_results[category] = (percentage_exact_match, average_partial_match)

        # Save the detailed results to a CSV file
        output_path = f'evaluation_results_{category.lower()}.csv'
        category_data.to_csv(output_path, index=False)
        print(f'Results for {category} saved to {output_path}')

        # Print the results
        print(f'{category} Exact Match Percentage: {percentage_exact_match:.2f}%')
        print(f'{category} Partial Match Percentage: {average_partial_match:.2f}%')

    # Plotting results
    fig, ax = plt.subplots()
    index = range(len(categories))
    bar_width = 0.35

    exact_matches = [final_results[cat][0] for cat in categories]
    partial_matches = [final_results[cat][1] for cat in categories]

    rects1 = ax.bar(index, exact_matches, bar_width, color='orange', label='Exact Matches')
    rects2 = ax.bar([p + bar_width for p in index], partial_matches, bar_width, color='purple', label='Partial Matches')

    ax.set_xlabel('Length Category')
    ax.set_ylabel('Matching Percentage')
    ax.set_title('Match Percentage by Query Length')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(categories)
    ax.legend()

    # Save the plot to a file
    plt.savefig('match_percentages_by_length.png')
    #plt.show()
