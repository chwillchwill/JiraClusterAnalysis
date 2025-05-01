import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_text(text, preserve_punctuation=False):
    """
    Clean and normalize text for better similarity detection

    Args:
        text: Text to clean
        preserve_punctuation: If True, keep punctuation for better sensitivity
    """
    if not isinstance(text, str):
        return str(text)

    # Convert to lowercase
    text = str(text).lower()

    if not preserve_punctuation:
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
    else:
        # Keep some punctuation but normalize spaces around them
        text = re.sub(r'([.,!?:;])', r' \1 ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def combine_row_data(row, sensitivity=0.5):
    """
    Combine all column values into a single string for analysis

    Args:
        row: Row to combine
        sensitivity: Higher sensitivity gives more weight to rare terms
    """
    # For higher sensitivity, we might want to repeat important columns
    if sensitivity > 0.7:
        # Try to identify key columns by their names
        potential_key_columns = ['description', 'title', 'name', 'category', 'text', 'summary']

        combined = []
        for col_name, val in row.items():
            # Check if column name contains any key terms
            is_key_column = any(key in str(col_name).lower() for key in potential_key_columns)

            # Repeat important columns based on sensitivity
            if is_key_column:
                # Add column value multiple times for emphasis
                repeat_count = 2 if sensitivity < 0.9 else 3
                combined.extend([str(val)] * repeat_count)
            else:
                combined.append(str(val))

        return ' '.join(combined)
    else:
        # Default behavior for lower sensitivity
        return ' '.join(str(val) for val in row)


def cluster_rows(data, header, n_clusters=None, sensitivity=0.5):
    """
    Cluster rows based on similarity across all columns.

    Args:
        data: DataFrame without header
        header: Original header row
        n_clusters: Number of clusters (optional)
        sensitivity: Clustering sensitivity (0.1-1.0, higher means more clusters)

    Returns:
        Reordered DataFrame with header and blank rows between clusters
    """
    logging.info("Preparing data for clustering...")

    # Combine all column data for each row into a single text string
    combined_texts = data.apply(lambda row: combine_row_data(row, sensitivity), axis=1)

    # Clean combined texts
    preserve_punctuation = sensitivity > 0.6  # Preserve punctuation for higher sensitivity
    cleaned_texts = [clean_text(text, preserve_punctuation) for text in combined_texts]

    # Convert text to numerical features using TF-IDF with sensitivity adjustments
    logging.info("Converting text to TF-IDF features...")

    # Adjust TF-IDF parameters based on sensitivity
    # Lower min_df and higher max_features for higher sensitivity
    min_df_value = max(0.005, 0.02 - (sensitivity * 0.015))  # Lower with higher sensitivity
    max_df_value = min(0.99, 0.90 + (sensitivity * 0.09))  # Higher with higher sensitivity

    vectorizer = TfidfVectorizer(
        min_df=min_df_value,  # Adjusted by sensitivity
        max_df=max_df_value,  # Adjusted by sensitivity
        stop_words='english',
        ngram_range=(1, 2 if sensitivity > 0.6 else 1)  # Use bigrams for higher sensitivity
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

        # Check if we have enough data for meaningful clustering
        if tfidf_matrix.shape[0] < 2:
            logging.warning("Not enough data for clustering. Returning original data.")
            return pd.DataFrame([header] + data.values.tolist())

        # Calculate similarity matrix
        logging.info("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # If n_clusters is not specified, estimate a reasonable number
        if n_clusters is None:
            # Use a heuristic: sqrt(n/2) as a starting point, adjusted by sensitivity
            n_clusters = max(2, int(np.sqrt(len(data) / 2) * (1 + sensitivity)))
            logging.info(f"Automatically set number of clusters to {n_clusters} (sensitivity: {sensitivity})")

        # Adjust linkage method based on sensitivity
        # 'ward' creates more evenly sized clusters
        # 'complete' is more sensitive to outliers and creates more varied clusters
        # 'average' is in between
        if sensitivity < 0.3:
            linkage_method = 'ward'
        elif sensitivity > 0.7:
            linkage_method = 'complete'
        else:
            linkage_method = 'average'

        # Perform hierarchical clustering with sensitivity adjustments
        logging.info(f"Performing clustering with {n_clusters} clusters using {linkage_method} linkage...")

        if linkage_method == 'ward':
            # Ward requires Euclidean distance, not precomputed
            # Convert similarity to Euclidean distance equivalent
            from sklearn.metrics import pairwise_distances
            # Use subset of features for higher dimensions if data is large
            if tfidf_matrix.shape[1] > 1000 and sensitivity > 0.5:
                from sklearn.decomposition import TruncatedSVD
                n_components = min(100, tfidf_matrix.shape[0] - 1)
                svd = TruncatedSVD(n_components=n_components)
                reduced_features = svd.fit_transform(tfidf_matrix)
                distances = pairwise_distances(reduced_features, metric='euclidean')
            else:
                distances = pairwise_distances(tfidf_matrix.toarray(), metric='euclidean')

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            clustering.fit(distances)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                distance_threshold=None
            )
            # Use 1 - similarity as distance
            clustering.fit(1 - similarity_matrix)

        # Use 1 - similarity as distance
        clustering.fit(1 - similarity_matrix)

        # Add cluster labels to the data
        data['cluster'] = clustering.labels_

        # Sort by cluster
        data = data.sort_values('cluster')

        # Create result with header as first row
        result = [header]

        # Add rows, with blank rows between different clusters
        current_cluster = None

        for _, row in data.iterrows():
            cluster_id = row['cluster']
            row_data = row.drop('cluster').tolist()

            # Insert blank row between different clusters (except before the first cluster)
            if current_cluster is not None and cluster_id != current_cluster:
                # Add a blank row (same number of columns as the data)
                result.append([''] * len(header))

            result.append(row_data)
            current_cluster = cluster_id

        return pd.DataFrame(result)

    except Exception as e:
        logging.error(f"Error during clustering: {str(e)}")
        logging.info("Falling back to original data order")
        return pd.DataFrame([header] + data.values.tolist())


def process_csv(input_file, output_file, n_clusters=None, sensitivity=0.5):
    """
    Read CSV, cluster similar rows, and write output.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        n_clusters: Number of clusters (optional)
        sensitivity: Clustering sensitivity (0.1-1.0, higher means more clusters)
    """
    try:
        logging.info(f"Reading input file: {input_file}")
        # Read the first line to get the delimiter
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        # Try to detect the delimiter
        if ',' in first_line:
            delimiter = ','
        elif ';' in first_line:
            delimiter = ';'
        elif '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','  # Default to comma

        # Read CSV file
        df = pd.read_csv(input_file, delimiter=delimiter, encoding='utf-8')

        # Extract header and data
        header = list(df.columns)
        data = df.copy()

        # If n_clusters is not specified, calculate based on sensitivity
        if n_clusters is None:
            # Base number of clusters calculation with sensitivity factor
            base_clusters = int(np.sqrt(len(data) / 2))
            # Adjust by sensitivity: higher sensitivity means more clusters
            n_clusters = max(2, int(base_clusters * (1 + sensitivity)))
            logging.info(f"Auto-calculated {n_clusters} clusters with sensitivity {sensitivity}")

        # Perform clustering
        logging.info("Clustering rows...")
        result_df = cluster_rows(data, header, n_clusters, sensitivity)

        # Write result to output file
        logging.info(f"Writing output to: {output_file}")
        result_df.to_csv(output_file, index=False, header=False, quoting=1)
        logging.info("Processing complete!")

        return True

    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        return False


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Cluster similar rows in a CSV file.')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--clusters', type=int, default=None, help='Number of clusters (optional)')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Clustering sensitivity (0.1-1.0, higher means more clusters)')

    args = parser.parse_args()

    # Validate sensitivity range
    if args.sensitivity < 0.1 or args.sensitivity > 1.0:
        print("Sensitivity must be between 0.1 and 1.0")
        return

    success = process_csv(args.input_file, args.output_file, args.clusters, args.sensitivity)

    if success:
        print(f"Successfully processed {args.input_file} and wrote results to {args.output_file}")
        print(f"Used sensitivity level: {args.sensitivity}")
    else:
        print("Processing failed. Check logs for details.")


if __name__ == "__main__":
    main()