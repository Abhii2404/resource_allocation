import pandas as pd
import ast
import numpy as np

def parse_resource_string(res_str):
    """Parses strings like "{'cpus': 0.02, 'memory': 0.01}" into a dict."""
    try:
        if pd.isna(res_str):
            return {'cpus': 0.0, 'memory': 0.0}
        # The dataset uses None which python eval handles, but ast needs careful handling
        # Replacing 'None' with '0' for safety in resource math
        cleaned = res_str.replace('None', '0') 
        return ast.literal_eval(cleaned)
    except:
        return {'cpus': 0.0, 'memory': 0.0}

def load_and_process_data(filepath, num_samples=1000):
    """
    Loads the dataset and prepares features for the Neural Network.
    We limit samples to speed up the demonstration.
    """
    df = pd.read_csv(filepath, nrows=num_samples)
    
    # 1. Parse Resources
    df['parsed_req'] = df['resource_request'].apply(parse_resource_string)
    df['parsed_avg'] = df['average_usage'].apply(parse_resource_string)
    
    # 2. Extract Numerical Features needed for Eq (1)
    # Execution Time (ET) = end_time - start_time
    df['ET'] = df['end_time'] - df['start_time']
    
    # Cost (C) = Assigned Memory (proxy from paper)
    df['C'] = df['assigned_memory']
    
    # System Efficiency (SE) = Average Usage / Resource Request (Simple ratio)
    def calc_efficiency(row):
        req_cpu = row['parsed_req'].get('cpus', 1)
        avg_cpu = row['parsed_avg'].get('cpus', 0)
        if req_cpu == 0: return 0
        return avg_cpu / req_cpu

    df['SE'] = df.apply(calc_efficiency, axis=1)
    
    # Response Time (RT) proxy: simple assumption based on wait time or duration
    df['RT'] = df['ET'] * 1.1 # Simulated slight delay
    
    # Normalize data (Crucial for Neural Networks)
    cols_to_norm = ['ET', 'C', 'SE', 'RT']
    for col in cols_to_norm:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        df[col] = df[col].fillna(0)
        
    return df

def get_10_test_tasks(df):
    """Selects 10 representative tasks for the final graph evaluation."""
    return df.iloc[0:10].copy().reset_index(drop=True)