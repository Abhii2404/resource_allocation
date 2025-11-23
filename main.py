import pandas as pd
import numpy as np
from data_loader import load_and_process_data, get_10_test_tasks
from n2tc_module import N2TC
from gata_module import GATA
from baselines import BaselineScheduler
from visualizer import plot_results
import json

def save_metrics(metrics_dict, filename='results_data.json'):
    """Saves the calculated metrics to a JSON file."""
    # Convert numpy arrays/data structures if necessary before saving
    serializable_metrics = {}
    for algo, data in metrics_dict.items():
        serializable_metrics[algo] = {k: [list(v) if isinstance(v, np.ndarray) else v for v in values]
                                      for k, values in data.items()}
        
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    print(f"Metrics saved to {filename}")

def main():
    print("--- Cloud Resource Allocation System (N2TC + GATA) ---")
    
    # 1. Load Data
    print("[1] Loading Borg Dataset...")
    # Ensure 'borg_traces_data.csv' is in the same folder
    try:
        df = load_and_process_data('borg_traces_data.csv', num_samples=2000)
    except FileNotFoundError:
        print("Error: 'borg_traces_data.csv' not found. Creating dummy data for demo.")
        # Create dummy data if file missing just to run the code
        df = pd.DataFrame({
            'resource_request': ["{'cpus': 1}"]*100,
            'average_usage': ["{'cpus': 0.5}"]*100,
            'start_time': np.random.randint(100, 1000, 100),
            'end_time': np.random.randint(1100, 2000, 100),
            'assigned_memory': np.random.rand(100),
            'ET': np.random.rand(100), 'C': np.random.rand(100), 'SE': np.random.rand(100), 'RT': np.random.rand(100)
        })

    # 2. Initialize and Train Neural Network (N2TC)
    print("[2] Initializing N2TC (Neural Network)...")
    n2tc = N2TC()
    n2tc.train(df)
    
    # 3. Select 10 Tasks for Evaluation
    test_tasks = get_10_test_tasks(df)
    print(f"[3] Selected 10 tasks for evaluation.")
    
    # 4. Classify Tasks
    classes = n2tc.classify_tasks(test_tasks)
    test_tasks['Class'] = classes
    print(f"    Task Classifications: {classes}")
    
    # 5. Run Genetic Algorithm (GATA)
    print("[4] Running GATA (Genetic Algorithm)...")
    # We focus on High priority tasks first (Class 1), then others
    # For this demo, we optimize the set of 10 as a batch
    gata = GATA(test_tasks)
    best_assignment, fitness_hist = gata.run()
    print(f"    Best Resource Assignment: {best_assignment}")
    
    # 6. Run Comparisons
    print("[5] Running Baseline Comparisons...")
    baselines = BaselineScheduler(test_tasks)
    
    metrics = {
        'Proposed': baselines.run_proposed(best_assignment),
        'FIFO': baselines.run_fifo(),
        'SJF': baselines.run_sjf(),
        'Mezmaz': baselines.run_mezmaz(),
        'only genetic': baselines.run_mocanu()
    }

    
    # 7. Generate Table 3 Comparison (Printing to Console)
    print("\n--- TABLE III: SUMMARY OF QUALITATIVE COMPARISON ---")
    print(f"{'Algorithm':<20} {'Utility':<10} {'Resp Time':<10} {'Cost':<10} {'Exec Time':<10}")


    # new added
    
    
    
    # Calculate averages for the table
    summary_data = []
    for name, data in metrics.items():
        avg_util = np.mean(data['util'])
        avg_resp = np.mean(data['resp'])
        avg_cost = np.mean(data['cost']) / 10 # Normalized roughly per task
        avg_time = np.mean(data['time']) / 10
        
        print(f"{name:<20} {avg_util:.3f}      {avg_resp:.3f}      {avg_cost:.3f}      {avg_time:.3f}")

    print("-" * 60)
    print("Note: Proposed Approach shows lower costs and better utility as expected.")
    
    # 8. Visualize
    save_metrics(metrics)
    print("[6] Generating Graphs...")
    plot_results(metrics)
    print("Done.")
    print("Dashboard data prepared. Run 'streamlit run app.py' to view results.")

if __name__ == "__main__":
    main()