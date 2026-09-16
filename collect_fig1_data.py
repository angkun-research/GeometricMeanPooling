import os
import json
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= CONFIGURATION =================
RESULTS_FILE = "./plots/fig1_results.json"
NUM_WORKERS = 4
GPUS = [0, 1]  # Available GPU IDs

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
DATASETS = ["mnist", "fashion_mnist", "cifar10"]
CONFIGS = ["Global", "Local", "All"]
POOLINGS = ["geo", "max", "avg"]
# =================================================

def get_all_tasks():
    tasks = []
    for dataset in DATASETS:
        for config in CONFIGS:
            for pooling in POOLINGS:
                for seed in SEEDS:
                    tasks.append({
                        "dataset": dataset,
                        "config": config,
                        "pooling": pooling,
                        "seed": seed
                    })
    return tasks

def run_single_task(task, gpu_id):
    """
    Executes a single training run as a subprocess.
    Captures the return value (best validation accuracy).
    """
    dataset = task["dataset"]
    config = task["config"]
    pooling = task["pooling"]
    seed = task["seed"]

    # Construct the python execution string to call the specific function and print result
    if dataset == "mnist":
        func_map = {"Global": ("run_mnist_config1", "run_config1"),
                    "Local": ("run_mnist_config2", "run_config2"),
                    "All": ("run_mnist_config3", "run_config3")}
        module, func = func_map[config]
        cmd = f"from {module} import {func}; print({func}(pooling='{pooling}', seed={seed}, epochs=200))"
    elif dataset == "fashion_mnist":
        model_class = f"DeepCNN{config}"
        cmd = f"from run_fashion_mnist import run_experiment, {model_class}; print(run_experiment({model_class}, pooling='{pooling}', seed={seed}, epochs=200))"
    elif dataset == "cifar10":
        model_class = f"CIFARCNN{config}"
        cmd = f"from run_cifar10 import run_experiment, {model_class}; print(run_experiment({model_class}, pooling='{pooling}', seed={seed}, epochs=200))"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Run as subprocess to isolate CUDA contexts and memory
        process = subprocess.Popen(
            ["python", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        stdout, stderr = process.communicate()

        # The runner functions return the accuracy and we print it in the command string
        # We look for the last float printed in stdout
        import re
        matches = re.findall(r"([0-9]*\.[0-9]+)", stdout)
        if matches:
            return float(matches[-1])
        else:
            print(f"Error parsing output for {task}: {stderr}")
            return None
    except Exception as e:
        print(f"Exception running task {task}: {e}")
        return None

def save_result(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

def main():
    tasks = get_all_tasks()

    # Load existing results for resumability
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    # Filter out completed tasks
    pending_tasks = []
    for t in tasks:
        key = f"{t['dataset']}_{t['config']}_{t['pooling']}_{t['seed']}"
        if key not in results:
            pending_tasks.append((key, t))

    print(f"Starting data collection for Figure 1.")
    print(f"Total tasks to run: {len(tasks)} | Pending tasks: {len(pending_tasks)}")
    print(f"Using {NUM_WORKERS} workers across GPUs {GPUS}")

    # Use a ProcessPoolExecutor for the worker pool
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Assign GPU ID based on worker index (Worker 0,1 -> GPU 0; Worker 2,3 -> GPU 1)
        # Since we don't know which worker gets which task, we can't easily map.
        # Instead, we use a simple loop to submit tasks and track them.

        # To properly manage GPUs, we'll maintain a list of active futures per GPU
        # but ProcessPoolExecutor handles the queue.
        # A simpler approach: use a semaphore or just assign based on current pending count.

        # Actually, for simplicity and reliability with subprocesses,
        # we can just map tasks to GPUs in batches of NUM_WORKERS.

        futures = {}
        task_idx = 0
        while task_idx < len(pending_tasks):
            # Submit up to NUM_WORKERS tasks
            for i in range(NUM_WORKERS):
                if task_idx >= len(pending_tasks):
                    break

                key, task = pending_tasks[task_idx]
                # Assign GPU: Worker 0,1 -> GPU 0; Worker 2,3 -> GPU 1
                # We approximate this by rotating through the GPUs based on submission index
                gpu_id = GPUS[(task_idx // 2) % len(GPUS)] if NUM_WORKERS == 4 else GPUS[task_idx % len(GPUS)]

                future = executor.submit(run_single_task, task, gpu_id)
                futures[future] = (key, task)
                task_idx += 1

            # Wait for at least one task to finish before submitting more to avoid overloading the queue
            # though ProcessPoolExecutor already limits concurrent workers.
            # Let's just submit all and use as_completed.

        for future in as_completed(futures):
            key, task = futures[future]
            try:
                acc = future.result()
                if acc is not None:
                    results[key] = acc
                    save_result(results) # Immediate persistence
                    print(f"Completed {key}: Acc={acc:.4f} | Progress: {len(results)}/{len(tasks)}")
                else:
                    print(f"Task failed for {key}")
            except Exception as e:
                print(f"Future raised exception for {key}: {e}")

    print("\nData collection complete.")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
        # to run without interruption: nohup python collect_fig1_data.py > plots/fig1_collection.log 2>&1 &