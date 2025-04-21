import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
import matplotlib
# --- FIX: Force Matplotlib to use a non-GUI backend ---
# This must be done BEFORE importing pyplot
matplotlib.use('Agg')
# --- End Fix ---
import matplotlib.pyplot as plt
import numpy as np
import io
import time # Import time for measuring execution

# --- Data Loading ---
# Assuming 'ratings_small.csv' is in the same directory as the script
file_path = 'ratings_small.csv'

print("--- Loading Data ---")
try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    print("UTF-8 decoding failed, trying latin1 encoding.")
    df = pd.read_csv(file_path, encoding='latin1')

print("Columns in the CSV:", df.columns)
# Basic check for expected columns
expected_columns = ['userId', 'movieId', 'rating', 'timestamp']
if list(df.columns) != expected_columns:
     print(f"Warning: Column names {list(df.columns)} do not match expected {expected_columns}. Adjust code if needed.")
     # Attempt to proceed assuming the first three columns are user, item, rating in order
     df_load = df.iloc[:, [0, 1, 2]]
     df_load.columns = ['userId', 'movieId', 'rating'] # Rename for consistency
else:
     df_load = df[['userId', 'movieId', 'rating']] # Select only necessary columns

# Define the reader object. Dynamically set rating scale.
reader = Reader(rating_scale=(df_load['rating'].min(), df_load['rating'].max()))
# Load data from the DataFrame
data = Dataset.load_from_df(df_load, reader)
print("Data loaded successfully into Surprise format.")
print("-" * 30)

# --- Part e: Similarity Metric Comparison ---
print("\n--- Part e: Evaluating Similarity Metrics ---")
similarity_metrics = ['cosine', 'msd', 'pearson']
results_similarity = {'User-based CF': {}, 'Item-based CF': {}}

for sim_metric in similarity_metrics:
    print(f"Evaluating similarity: {sim_metric}")
    sim_options = {'name': sim_metric, 'user_based': True}
    algo_user = KNNBasic(sim_options=sim_options)
    print("  Running User-based CF...")
    start_time = time.time()
    cv_user = cross_validate(algo_user, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=-1)
    end_time = time.time()
    print(f"  User-based CF ({sim_metric}) completed in {end_time - start_time:.2f} seconds.")
    results_similarity['User-based CF'][sim_metric] = {
        'MAE': cv_user['test_mae'].mean(),
        'RMSE': cv_user['test_rmse'].mean()
    }

    sim_options = {'name': sim_metric, 'user_based': False}
    algo_item = KNNBasic(sim_options=sim_options)
    print("  Running Item-based CF...")
    start_time = time.time()
    cv_item = cross_validate(algo_item, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=-1)
    end_time = time.time()
    print(f"  Item-based CF ({sim_metric}) completed in {end_time - start_time:.2f} seconds.")
    results_similarity['Item-based CF'][sim_metric] = {
        'MAE': cv_item['test_mae'].mean(),
        'RMSE': cv_item['test_rmse'].mean()
    }

# Prepare data for plotting
user_mae = [results_similarity['User-based CF'][m]['MAE'] for m in similarity_metrics]
user_rmse = [results_similarity['User-based CF'][m]['RMSE'] for m in similarity_metrics]
item_mae = [results_similarity['Item-based CF'][m]['MAE'] for m in similarity_metrics]
item_rmse = [results_similarity['Item-based CF'][m]['RMSE'] for m in similarity_metrics]

print("\nSimilarity Metrics Results:")
print("User-based CF:", results_similarity['User-based CF'])
print("Item-based CF:", results_similarity['Item-based CF'])

# Plotting Similarity Results
fig_sim, axs_sim = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig_sim.suptitle('Impact of Similarity Metrics (5-fold CV Averages)')

# MAE Plot
axs_sim[0].plot(similarity_metrics, user_mae, marker='o', linestyle='-', label='User-based CF')
axs_sim[0].plot(similarity_metrics, item_mae, marker='x', linestyle='--', label='Item-based CF')
axs_sim[0].set_title('Mean Absolute Error (MAE)')
axs_sim[0].set_xlabel('Similarity Metric')
axs_sim[0].set_ylabel('Average MAE')
axs_sim[0].grid(True)
axs_sim[0].legend()

# RMSE Plot
axs_sim[1].plot(similarity_metrics, user_rmse, marker='o', linestyle='-', label='User-based CF')
axs_sim[1].plot(similarity_metrics, item_rmse, marker='x', linestyle='--', label='Item-based CF')
axs_sim[1].set_title('Root Mean Squared Error (RMSE)')
axs_sim[1].set_xlabel('Similarity Metric')
axs_sim[1].set_ylabel('Average RMSE')
axs_sim[1].grid(True)
axs_sim[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig('similarity_comparison.png') # Save the plot
print("\nSimilarity comparison plot saved as 'similarity_comparison.png'")
# plt.show() # Still commented out, not needed when saving file

# Consistency Check
print("\nConsistency Check:")
# Find best metric for each type based on RMSE
best_sim_user_rmse = min(results_similarity['User-based CF'], key=lambda k: results_similarity['User-based CF'][k]['RMSE'])
best_sim_item_rmse = min(results_similarity['Item-based CF'], key=lambda k: results_similarity['Item-based CF'][k]['RMSE'])

print(f"  Best similarity for User CF (RMSE): {best_sim_user_rmse}")
print(f"  Best similarity for Item CF (RMSE): {best_sim_item_rmse}")

if best_sim_user_rmse == best_sim_item_rmse:
    print("  The impact of similarity metrics appears consistent for User-based and Item-based CF (based on RMSE).")
else:
    print("  The impact of similarity metrics is NOT consistent for User-based and Item-based CF (based on RMSE).")
print("-" * 30)


# --- Part f & g: Neighbor Count (K) Impact ---
print("\n--- Part f & g: Evaluating Number of Neighbors (K) ---")
# Using MSD similarity as an example, as it often performs well. Change if needed.
chosen_similarity = 'msd'
print(f"Using '{chosen_similarity}' similarity for neighbor analysis.")

k_values = range(5, 51, 5) # Test K from 5 to 50 in steps of 5
results_k_user = {}
results_k_item = {}

for k in k_values:
    print(f"Evaluating K = {k}")
    # User-based CF
    sim_options_user = {'name': chosen_similarity, 'user_based': True}
    algo_user = KNNBasic(k=k, sim_options=sim_options_user)
    print(f"  Running User-based CF (k={k})...")
    start_time = time.time()
    cv_user = cross_validate(algo_user, data, measures=['RMSE'], cv=5, verbose=False, n_jobs=-1)
    end_time = time.time()
    print(f"  User-based CF (k={k}) completed in {end_time - start_time:.2f} seconds.")
    results_k_user[k] = cv_user['test_rmse'].mean()

    # Item-based CF
    sim_options_item = {'name': chosen_similarity, 'user_based': False}
    algo_item = KNNBasic(k=k, sim_options=sim_options_item)
    print(f"  Running Item-based CF (k={k})...")
    start_time = time.time()
    cv_item = cross_validate(algo_item, data, measures=['RMSE'], cv=5, verbose=False, n_jobs=-1)
    end_time = time.time()
    print(f"  Item-based CF (k={k}) completed in {end_time - start_time:.2f} seconds.")
    results_k_item[k] = cv_item['test_rmse'].mean()

# Find best K for each
best_k_user = min(results_k_user, key=results_k_user.get)
min_rmse_user = results_k_user[best_k_user]
best_k_item = min(results_k_item, key=results_k_item.get)
min_rmse_item = results_k_item[best_k_item]

print("\nNeighbor Count Results (RMSE):")
print("User-based CF:", results_k_user)
print("Item-based CF:", results_k_item)

# Plotting K Results
plt.figure(figsize=(10, 6))
plt.plot(list(results_k_user.keys()), list(results_k_user.values()), marker='o', linestyle='-', label='User-based CF')
plt.plot(list(results_k_item.keys()), list(results_k_item.values()), marker='x', linestyle='--', label='Item-based CF')
plt.title(f'Impact of Number of Neighbors (K) on RMSE (Similarity: {chosen_similarity})')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Average RMSE (5-fold CV)')
plt.xticks(list(k_values)) # Ensure all tested K values are shown as ticks
plt.grid(True)
plt.legend()
# Highlight the best K values
plt.plot(best_k_user, min_rmse_user, 'ro', markersize=8, label=f'Best K (User) = {best_k_user}')
plt.plot(best_k_item, min_rmse_item, 'g^', markersize=8, label=f'Best K (Item) = {best_k_item}')
plt.legend() # Call legend again to include the best K markers
plt.tight_layout()
plt.savefig('neighbor_impact.png') # Save the plot
print("\nNeighbor impact plot saved as 'neighbor_impact.png'")
# plt.show() # Still commented out

# --- Part g: Best K Comparison ---
print("\n--- Part g: Best K Identification ---")
print(f"Best K for User-based CF: {best_k_user} (RMSE: {min_rmse_user:.4f})")
print(f"Best K for Item-based CF: {best_k_item} (RMSE: {min_rmse_item:.4f})")

if best_k_user == best_k_item:
    print("\nThe best K value is the same for both User-based and Item-based collaborative filtering.")
else:
    print("\nThe best K value is different for User-based and Item-based collaborative filtering.")
print("-" * 30)

print("\nAnalysis complete.")
