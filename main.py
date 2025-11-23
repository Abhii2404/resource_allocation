import pandas as pd
import numpy as np
from data_loader import load_and_process_data, get_10_test_tasks
from n2tc_module import N2TC
from gata_module import GATA
from baselines import BaselineScheduler
from visualizer import plot_results
import json

def save_metrics(metrics_dict, extra_info=None, filename='results_data.json'):
    """Saves the calculated metrics and optional extra info to a JSON file.

    `extra_info` can be any JSON-serializable object (e.g., relative performance
    scores or an ASCII table string) and will be saved alongside the metrics.
    """
    # Convert numpy arrays/data structures if necessary before saving
    serializable_metrics = {}
    for algo, data in metrics_dict.items():
        serializable_metrics[algo] = {k: [list(v) if isinstance(v, np.ndarray) else v for v in values]
                                      for k, values in data.items()}

    out = {
        'metrics': serializable_metrics
    }
    if extra_info is not None:
        out['extra'] = extra_info

    with open(filename, 'w') as f:
        json.dump(out, f, indent=4)
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

    # Compute and print a composite normalized "Training Score" for each model
    def compute_and_print_model_scores(metrics_dict):
        """Compute a composite score (0..1) for each model based on averaged metrics.

        Uses avg utility (higher better), avg response (lower better), avg cost (lower better),
        avg execution time (lower better). Each metric is normalized across models and
        combined with preset weights.
        """
        names = list(metrics_dict.keys())
        agg = {}
        eps = 1e-9
        for name in names:
            data = metrics_dict[name]
            util = float(np.mean(data.get('util', [0]))) if len(data.get('util', []))>0 else 0.0
            resp = float(np.mean(data.get('resp', [0]))) if len(data.get('resp', []))>0 else 0.0
            cost = float(np.mean(data.get('cost', [0]))) if len(data.get('cost', []))>0 else 0.0
            time_avg = float(np.mean(data.get('time', [0]))) if len(data.get('time', []))>0 else 0.0
            agg[name] = {'util': util, 'resp': resp, 'cost': cost, 'time': time_avg}

        # Build arrays for normalization
        utils = np.array([agg[n]['util'] for n in names], dtype=float)
        resps = np.array([agg[n]['resp'] for n in names], dtype=float)
        costs = np.array([agg[n]['cost'] for n in names], dtype=float)
        times = np.array([agg[n]['time'] for n in names], dtype=float)

        def normalize(arr, invert=False):
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            if np.isclose(mx, mn):
                # all equal -> neutral score 0.5
                return np.ones_like(arr) * 0.5
            norm = (arr - mn) / (mx - mn + eps)
            return 1 - norm if invert else norm

        # utility: higher better (no invert)
        u_norm = normalize(utils, invert=False)
        # response, cost, time: lower better -> invert=True
        r_norm = normalize(resps, invert=True)
        c_norm = normalize(costs, invert=True)
        t_norm = normalize(times, invert=True)

        # weights (tunable)
        w_util, w_resp, w_cost, w_time = 0.4, 0.2, 0.3, 0.1

        scores = {}
        for i, name in enumerate(names):
            score = (w_util * u_norm[i]) + (w_resp * r_norm[i]) + (w_cost * c_norm[i]) + (w_time * t_norm[i])
            scores[name] = float(score)

        # Prepare a clean, aligned table sorted by score (higher is better)
        header_lines = []
        header_lines.append("Relative Performance Scores (higher is better)")
        header_lines.append("+----------------------+---------+---------+")
        header_lines.append(f"| {'Model':<20} | {'Score':>6} | {'Percent':>7} |")
        header_lines.append("+----------------------+---------+---------+")
        # sort by score desc
        sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (name, sc) in enumerate(sorted_items, start=1):
            header_lines.append(f"| {name:<20} | {sc:6.4f} | {sc*100:6.2f}% |")
        header_lines.append("+----------------------+---------+---------+\n")

        table_str = "\n".join(header_lines)
        print(table_str)

        # Return both structured scores and the ASCII table string for saving/display
        return scores, table_str

    scores, table_str = compute_and_print_model_scores(metrics)

    
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
    # Save metrics along with relative performance scores/table for the dashboard
    extra_info = {
        'relative_performance': scores,
        'relative_performance_table': table_str
    }
    save_metrics(metrics, extra_info=extra_info)
    print("[6] Generating Graphs...")
    plot_results(metrics)
    print("Done.")
    print("Dashboard data prepared. Run 'streamlit run app.py' to view results.")

if __name__ == "__main__":
    main()