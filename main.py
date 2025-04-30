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


def clean_text(text):
    """Clean and normalize text for better similarity detection"""
    if not isinstance(text, str):
        return str(text)
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def combine_row_data(row):
    """Combine all column values into a single string for analysis"""
    return ' '.join(str(val) for val in row)


def cluster_rows(data, header, n_clusters=None):
    """
    Cluster rows based on similarity across all columns.
    
    Args:
        data: DataFrame without header
        header: Original header row
        n_clusters: Number of clusters (optional)
    
    Returns:
        Reordered DataFrame with header and blank rows between clusters
    """
    logging.info("Preparing data for clustering...")
    
    # Combine all column data for each row into a single text string
    combined_texts = data.apply(combine_row_data, axis=1)
    
    # Clean combined texts
    cleaned_texts = [clean_text(text) for text in combined_texts]
    
    # Convert text to numerical features using TF-IDF
    logging.info("Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(
        min_df=0.01,  # Ignore terms that appear in less than 1% of documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        stop_words='english'
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
            # Use a heuristic: sqrt(n/2) as a starting point
            n_clusters = max(2, int(np.sqrt(len(data) / 2)))
            logging.info(f"Automatically set number of clusters to {n_clusters}")
        
        # Perform hierarchical clustering
        logging.info(f"Performing clustering with {n_clusters} clusters...")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='average',
            distance_threshold=None
        )
        
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


def process_csv(input_file, output_file, n_clusters=None):
    """
    Read CSV, cluster similar rows, and write output.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        n_clusters: Number of clusters (optional)
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
        
        # Perform clustering
        logging.info("Clustering rows...")
        result_df = cluster_rows(data, header, n_clusters)
        
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
    
    args = parser.parse_args()
    
    success = process_csv(args.input_file, args.output_file, args.clusters)
    
    if success:
        print(f"Successfully processed {args.input_file} and wrote results to {args.output_file}")
    else:
        print("Processing failed. Check logs for details.")


if __name__ == "__main__":
    main()

