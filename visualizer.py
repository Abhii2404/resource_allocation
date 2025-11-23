import matplotlib.pyplot as plt

def plot_results(metrics_dict):
    """
    Generates Figures 5, 6, 7, 8 from the paper and saves them as PNGs.
    """
    tasks_x = list(range(1, 11))
    
    methods = ['Proposed', 'FIFO', 'SJF', 'Mezmaz', 'only genetic']
    styles = ['-d', '-s', '-^', '-x', '-*']
    colors = ['black', 'orange', 'gray', 'gold', 'lightgreen']
    
    # FIG 5: Execution Times
    plt.figure(figsize=(10, 6))
    for m, style, col in zip(methods, styles, colors):
        plt.plot(tasks_x, metrics_dict[m]['time'], style, color=col, label=m)
    plt.xlabel('Task')
    plt.ylabel('Time (s)')
    plt.title('Fig. 5: Execution times of 10 tasks')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # --- CHANGE IS HERE ---
    plt.savefig('Fig5_Execution_Time.png')
    plt.close() # Close the figure object to free memory

    # FIG 6: Utilization
    plt.figure(figsize=(10, 6))
    for m, style, col in zip(methods, styles, colors):
        plt.plot(tasks_x, metrics_dict[m]['util'], style, color=col, label=m)
    plt.xlabel('Task')
    plt.ylabel('Utility (%)')
    plt.title('Fig. 6: System Utilization Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # --- CHANGE IS HERE ---
    plt.savefig('Fig6_Utilization.png')
    plt.close()

    # FIG 7: Costs
    plt.figure(figsize=(10, 6))
    for m, style, col in zip(methods, styles, colors):
        plt.plot(tasks_x, metrics_dict[m]['cost'], style, color=col, label=m)
    plt.xlabel('Task')
    plt.ylabel('Cost ($)')
    plt.title('Fig. 7: Costs of executing tasks')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # --- CHANGE IS HERE ---
    plt.savefig('Fig7_Costs.png')
    plt.close()

    # FIG 8: Response Times
    plt.figure(figsize=(10, 6))
    for m, style, col in zip(methods, styles, colors):
        plt.plot(tasks_x, metrics_dict[m]['resp'], style, color=col, label=m)
    plt.xlabel('Task')
    plt.ylabel('Time (s)')
    plt.title('Fig. 8: Response Times')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # --- CHANGE IS HERE ---
    plt.savefig('Fig8_Response_Time.png')
    plt.close()