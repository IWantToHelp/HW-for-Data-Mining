import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.stats import mode as scipy_mode # Use scipy's mode for majority voting
from collections import Counter

# --- Configuration ---
# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- File Paths ---
DATA_FILE = 'data.csv'
LABEL_FILE = 'label.csv'

# --- Data Loading ---
print(f"Loading data from {DATA_FILE}...")
try:
    # Load the features, assuming no header row
    data = pd.read_csv(DATA_FILE, header=None)
    # Ensure data is non-negative for Generalized Jaccard (min/max assumes non-negative)
    # If data can be negative, the interpretation of Generalized Jaccard might need adjustment.
    # Clipping at 0 is a common approach if negative values are small noise.
    X = np.clip(data.values, 0, None)
    print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
    if np.any(X < 0):
         print("Warning: Data contained negative values, clipped to 0 for Generalized Jaccard.")

except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please make sure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

print(f"Loading labels from {LABEL_FILE}...")
try:
    labels_df = pd.read_csv(LABEL_FILE, header=None)
    true_labels = labels_df.values.flatten() # Flatten if it's a single column
    # Determine K from the number of unique labels
    K = len(np.unique(true_labels))
    print(f"Labels loaded successfully. Found K={K} unique labels.")
except FileNotFoundError:
    print(f"Error: {LABEL_FILE} not found. Cannot determine K or calculate accuracy.")
    exit()
except Exception as e:
    print(f"An error occurred while loading labels: {e}")
    exit()

# --- Data Preprocessing for Jaccard (Binary - Kept for reference, but not used by Generalized Jaccard) ---
# print("Binarizing data for standard Jaccard distance...")
# X_binary = (X > 0).astype(int) # This is NOT used for Generalized Jaccard
# print("Binarization complete.")

# --- Distance Functions ---

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points (vectors)."""
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))

def cosine_distance(point1, point2):
    """Calculates the Cosine distance between two points (vectors)."""
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)
    if norm1 == 0 or norm2 == 0:
        return 1.0 # Max distance if one vector is zero
    similarity = np.dot(point1, point2) / (norm1 * norm2)
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity

# def jaccard_distance(point1, point2): # Standard Binary Jaccard - kept for reference
#     """Calculates the Jaccard distance between two BINARY points (vectors)."""
#     point1 = np.asarray(point1).astype(bool)
#     point2 = np.asarray(point2).astype(bool)
#     intersection = np.sum(point1 & point2)
#     union = np.sum(point1 | point2)
#     if union == 0: return 0.0
#     similarity = intersection / union
#     return 1.0 - similarity

def generalized_jaccard_distance(point1, point2):
    """
    Calculates the Generalized Jaccard distance (1 - Ružička similarity)
    between two continuous, non-negative points (vectors).
    Formula: 1 - (sum(min(p1_i, p2_i)) / sum(max(p1_i, p2_i)))
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    # Ensure non-negativity (already done during loading, but good practice)
    # point1 = np.maximum(point1, 0)
    # point2 = np.maximum(point2, 0)

    min_sum = np.sum(np.minimum(point1, point2))
    max_sum = np.sum(np.maximum(point1, point2))

    if max_sum == 0:
        # This happens only if both vectors are all zeros
        return 0.0

    similarity = min_sum / max_sum
    # Distance is 1 - similarity
    return 1.0 - similarity


# --- Helper Functions ---

def get_distance_function(metric_name):
    """Returns the appropriate distance function based on the metric name."""
    if metric_name == 'euclidean':
        return euclidean_distance
    elif metric_name == 'cosine':
        return cosine_distance
    # elif metric_name == 'jaccard': # Standard binary Jaccard
    #     return jaccard_distance
    elif metric_name == 'generalized_jaccard': # Use Generalized Jaccard
        return generalized_jaccard_distance
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")

def initialize_centroids_random(X, k):
    """Initializes k centroids by randomly selecting k data points from X."""
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids, distance_func):
    """Assigns each data point in X to the nearest centroid using the specified distance function."""
    n_samples = X.shape[0]
    k = centroids.shape[0]
    cluster_assignments = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        distances = [distance_func(X[i], centroids[j]) for j in range(k)]
        nearest_centroid_index = np.argmin(distances)
        cluster_assignments[i] = nearest_centroid_index

    return cluster_assignments

def update_centroids(X, cluster_assignments, k):
    """
    Updates the centroids based on the mean of points in each cluster.
    This standard mean update is used for Euclidean, Cosine, and Generalized Jaccard here.
    """
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features))

    for j in range(k):
        points_in_cluster = X[cluster_assignments == j]

        if len(points_in_cluster) > 0:
            # Use the mean for all metrics in this implementation
            new_centroids[j] = np.mean(points_in_cluster, axis=0)
        else:
            # Handle empty clusters: Re-initialize centroid randomly from the dataset X
            print(f"Warning: Cluster {j} became empty. Re-initializing centroid.")
            new_centroids[j] = X[np.random.choice(X.shape[0])]

    return new_centroids

def calculate_sse(X, centroids, cluster_assignments, distance_func):
    """Calculates the Sum of Squared Errors (SSE) for the clustering."""
    sse = 0.0
    k = centroids.shape[0]
    for j in range(k):
        points_in_cluster = X[cluster_assignments == j]
        if len(points_in_cluster) > 0:
            centroid = centroids[j]
            sse += np.sum([distance_func(point, centroid)**2 for point in points_in_cluster])
    return sse

def calculate_accuracy(true_labels, cluster_assignments, k):
    """Calculates clustering accuracy using majority vote labeling."""
    n_samples = len(true_labels)
    cluster_labels = np.zeros(k, dtype=int)

    for j in range(k):
        labels_in_cluster = true_labels[cluster_assignments == j]
        if len(labels_in_cluster) > 0:
            mode_result = scipy_mode(labels_in_cluster)
            majority_label = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode
            cluster_labels[j] = majority_label
        else:
             cluster_labels[j] = -1 # Placeholder

    correct_predictions = 0
    for i in range(n_samples):
        assigned_cluster = cluster_assignments[i]
        if cluster_labels[assigned_cluster] == true_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / n_samples
    return accuracy

# --- K-Means Algorithm Implementation ---

def kmeans(X_input, k, distance_metric, max_iterations=100, tolerance=1e-4, stop_on_sse_increase=False):
    """
    Performs K-means clustering with specified distance metric and stop criteria.

    Args:
        X_input (np.ndarray): The dataset (n_samples, n_features).
                               Should be NON-NEGATIVE for 'generalized_jaccard'.
        k (int): The number of clusters.
        distance_metric (str): 'euclidean', 'cosine', or 'generalized_jaccard'.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Threshold for centroid movement convergence.
                           Set to 0 for exact match stop condition.
                           Set to infinity if only relying on other criteria.
        stop_on_sse_increase (bool): If True, stop if SSE increases.

    Returns:
        dict: A dictionary containing results.
    """
    start_time = time.time()
    print(f"\nStarting K-Means with metric='{distance_metric}', k={k}, max_iter={max_iterations}, tol={tolerance}, stop_on_sse_increase={stop_on_sse_increase}")

    distance_func = get_distance_function(distance_metric)
    n_samples, n_features = X_input.shape

    # 1. Initialize centroids
    centroids = initialize_centroids_random(X_input, k)
    print(f"Initial centroids shape: {centroids.shape}")

    cluster_assignments = np.zeros(n_samples, dtype=int)
    sse_history = []
    stop_reason = 'max_iterations'

    for iteration in range(max_iterations):
        old_centroids = np.copy(centroids)
        prev_sse = sse_history[-1] if sse_history else float('inf')

        # 2. Assign points to the nearest cluster
        cluster_assignments = assign_clusters(X_input, centroids, distance_func)

        # 3. Update centroids (using standard mean for all metrics here)
        centroids = update_centroids(X_input, cluster_assignments, k)

        # 4. Calculate current SSE
        current_sse = calculate_sse(X_input, centroids, cluster_assignments, distance_func)
        sse_history.append(current_sse)
        print(f"  Iter {iteration + 1}/{max_iterations}: SSE = {current_sse:.4f}")

        # 5. Check for stop criteria
        # a) SSE Increase
        if stop_on_sse_increase and current_sse > prev_sse and iteration > 0:
             print(f"  Stopping: SSE increased from {prev_sse:.4f} to {current_sse:.4f}")
             centroids = old_centroids
             cluster_assignments = assign_clusters(X_input, centroids, distance_func)
             sse_history.pop()
             stop_reason = 'sse_increase'
             break

        # b) Centroid Convergence (using Euclidean distance for shift magnitude)
        centroid_shift = np.sum([euclidean_distance(old_centroids[j], centroids[j]) for j in range(k)])
        print(f"  Centroid shift: {centroid_shift:.4f}")
        if centroid_shift <= tolerance:
            print(f"\nConvergence reached after {iteration + 1} iterations (shift <= {tolerance}).")
            stop_reason = 'convergence'
            break

    else: # No break occurred
        print(f"\nK-Means finished after reaching the maximum of {max_iterations} iterations.")
        stop_reason = 'max_iterations'

    end_time = time.time()
    time_taken = end_time - start_time
    final_sse = sse_history[-1] if sse_history else float('inf')

    print(f"Finished K-Means ({distance_metric}). Time: {time_taken:.2f}s, Iterations: {iteration + 1}, Final SSE: {final_sse:.4f}")

    return {
        'centroids': centroids,
        'assignments': cluster_assignments,
        'sse': final_sse,
        'iterations': iteration + 1,
        'time_taken': time_taken,
        'stop_reason': stop_reason,
        'sse_history': sse_history
    }

# --- Main Execution & Comparisons ---
if __name__ == "__main__":

    results = {}
    accuracies = {}
    metrics_to_run = ['euclidean', 'cosine', 'generalized_jaccard'] # Updated list

    # --- Q1, Q2, Q3 Execution ---
    print("\n--- Running K-Means for Q1, Q2, Q3 ---")
    MAX_ITER_Q3 = 500
    TOLERANCE_Q3 = 1e-6

    for metric in metrics_to_run:
        # Generalized Jaccard uses the original (non-negative) data X
        # Euclidean and Cosine also use X
        data_to_use = X
        results[metric] = kmeans(data_to_use, K, metric,
                                 max_iterations=MAX_ITER_Q3,
                                 tolerance=TOLERANCE_Q3,
                                 stop_on_sse_increase=True)
        accuracies[metric] = calculate_accuracy(true_labels, results[metric]['assignments'], K)

    # --- Q1: SSE Comparison ---
    print("\n--- Q1: SSE Comparison ---")
    sse_values = {metric: res['sse'] for metric, res in results.items()}
    print(f"Final SSE values (K={K}, MaxIter={MAX_ITER_Q3}, StopOnSSEIncrease=True, Tol={TOLERANCE_Q3}):")
    for metric, sse in sse_values.items():
        print(f"  - {metric.capitalize()}: {sse:.4f}")
    best_sse_metric = min(sse_values, key=sse_values.get)
    print(f"Based on SSE, '{best_sse_metric.capitalize()}' K-means performed best (lowest SSE).")

    # --- Q2: Accuracy Comparison ---
    print("\n--- Q2: Accuracy Comparison ---")
    print(f"Predictive Accuracy (using majority vote label):")
    for metric, acc in accuracies.items():
        print(f"  - {metric.capitalize()}: {acc:.4f} ({acc*100:.2f}%)")
    best_acc_metric = max(accuracies, key=accuracies.get)
    print(f"Based on Accuracy, '{best_acc_metric.capitalize()}' K-means performed best (highest accuracy).")

    # --- Q3: Convergence Comparison ---
    print("\n--- Q3: Convergence Comparison (Iterations & Time) ---")
    print(f"Iterations and Time Taken (Stop Criteria: No Change (Tol={TOLERANCE_Q3}) OR SSE Increase OR Max Iter={MAX_ITER_Q3}):")
    for metric, res in results.items():
        print(f"  - {metric.capitalize()}:")
        print(f"    - Iterations: {res['iterations']} ({res['stop_reason']})")
        print(f"    - Time Taken: {res['time_taken']:.2f} seconds")
    max_iter_metric = max(results, key=lambda m: results[m]['iterations'])
    max_time_metric = max(results, key=lambda m: results[m]['time_taken'])
    print(f"'{max_iter_metric.capitalize()}' required the most iterations.")
    print(f"'{max_time_metric.capitalize()}' took the most time.")


    # --- Q4: SSE Comparison w.r.t. Terminating Conditions ---
    print("\n--- Q4: SSE Comparison for Different Terminating Conditions ---")
    MAX_ITER_Q4 = 100
    sse_q4 = {metric: {} for metric in metrics_to_run}

    # Condition 1: No change in centroid position (or max iter)
    print(f"\nRunning for Condition 1: Stop on No Change (Tol={TOLERANCE_Q3}) or Max Iter ({MAX_ITER_Q4})...")
    for metric in metrics_to_run:
        data_to_use = X
        res = kmeans(data_to_use, K, metric,
                     max_iterations=MAX_ITER_Q4,
                     tolerance=TOLERANCE_Q3,
                     stop_on_sse_increase=False)
        sse_q4[metric]['no_change'] = res['sse']

    # Condition 2: SSE value increases
    print(f"\nRunning for Condition 2: Stop on SSE Increase or Max Iter ({MAX_ITER_Q4})...")
    for metric in metrics_to_run:
        data_to_use = X
        res = kmeans(data_to_use, K, metric,
                     max_iterations=MAX_ITER_Q4,
                     tolerance=float('inf'),
                     stop_on_sse_increase=True)
        sse_q4[metric]['sse_increase'] = res['sse']

    # Condition 3: Maximum preset value of iteration is complete
    print(f"\nRunning for Condition 3: Stop on Max Iter ({MAX_ITER_Q4})...")
    for metric in metrics_to_run:
        data_to_use = X
        res = kmeans(data_to_use, K, metric,
                     max_iterations=MAX_ITER_Q4,
                     tolerance=float('inf'),
                     stop_on_sse_increase=False)
        sse_q4[metric]['max_iter'] = res['sse']

    # Print Q4 results table
    print("\nSSE Comparison Table (Lower is Better):")
    print(f"{'Metric':<20} | {'Stop: No Change':<18} | {'Stop: SSE Increase':<20} | {'Stop: Max Iter':<18}")
    print("-" * 85)
    for metric in metrics_to_run:
        # Format metric name nicely for the table
        metric_display = metric.replace('_', ' ').capitalize()
        print(f"{metric_display:<20} | {sse_q4[metric]['no_change']:<18.4f} | {sse_q4[metric]['sse_increase']:<20.4f} | {sse_q4[metric]['max_iter']:<18.4f}")

    print("\n--- End of Analysis ---")

