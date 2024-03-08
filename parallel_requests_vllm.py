import requests
import time
import argparse
import functools
import concurrent.futures
from utils.dataset import load_conala, load_code_contests, load_dummy_data
from typing import Dict
import numpy as np
import pandas as pd


def warmup(warmup_requests=13):
    dummy_data = {
        "input": "Write fibonacci series code"
      }  # or any representative input
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(functools.partial(send_request, input_data=dummy_data)) for _ in range(warmup_requests)]
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()  # Discard results, just warming up


def send_request(input_data: Dict):
    url = 'http://localhost:7000/v2/models/vllm_model/generate'
    
    data = {"text_input": input_data['input'], 
            "parameters": 
                {
                "stream": False, 
                "temperature": 0,
                "max_tokens": 300,
                }
            }
    start_time = time.perf_counter()  # Start the timer
    response = requests.post(url, json=data)
    end_time = time.perf_counter()  # End the timer
    return response.status_code, end_time - start_time  # Return status code and time taken

def main():
    
    parser = argparse.ArgumentParser(description="Process some inputs.")
    # Add an argument for the dataset name
    parser.add_argument('--dataset_name', type=str, help='The name of the dataset')
    # Parse the command line arguments
    args = parser.parse_args()

    print("[INFO] Loading dataset")
    if args.dataset_name == "neulab/conala":
        data = load_conala()
    elif args.dataset_name == "deepmind/code_contests":
        data = load_code_contests()
    elif args.dataset_name == "dummy_data":
        data = load_dummy_data()
    else:
        print("This dataset is not implemented")
        exit()

    print("[INFO] Warming up GPU ...")
    warmup()
    
    print("[INFO] Running benchmark")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(functools.partial(send_request, input_data=input_data)) for input_data in data]

        # Collect results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Count successful and failed requests
        successful_requests = sum(1 for result in results if result[0] == 200)
        failed_requests = len(results) - successful_requests

        # Calculate average request time
        request_times = [result[1] for result in results]
        p50_time = np.percentile(request_times, 50)  # 50th percentile, median
        p90_time = np.percentile(request_times, 90)  # 90th percentile
        p99_time = np.percentile(request_times, 99)  # 99th percentile
        

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Successful Requests", "Failed Requests", "P50 Latency (seconds)", "P90 Latency (seconds)", "P99 Latency (seconds)"],
        "Value": [successful_requests, failed_requests,  f"{p50_time:.2f}", f"{p90_time:.2f}", f"{p99_time:.2f}"]
    })

    print(metrics_df)

    # Save the DataFrame to a CSV file
    sanitized_dataset = args.dataset_name.replace('/', '_')
    csv_file = f"benchmark_{sanitized_dataset}_vllm.csv"
    metrics_df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    main()
