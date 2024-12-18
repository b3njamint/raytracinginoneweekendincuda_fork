import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def compile_cuda_program(cuda_file='main.cu'):
    """Compile the CUDA program."""
    compile_command = f"nvcc {cuda_file} -o ray_tracing"
    result = subprocess.run(compile_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {result.stderr}")

def modify_cuda_source_object_count(max_a, max_b):
    """
    Modify the CUDA source file to change the number of objects created.
    
    Args:
        max_a (int): First loop limit
        max_b (int): Second loop limit
    """
    with open('main.cu', 'r') as f:
        source_code = f.read()
    
    # Replace loop limits in create_world function
    source_code = re.sub(r'for\(int a = -\d+; a < \d+; a\+\+\)', f'for(int a = -{max_a}; a < {max_a}; a++)', source_code)
    source_code = re.sub(r'for\(int b = -\d+; b < \d+; b\+\+\)', f'for(int b = -{max_b}; b < {max_b}; b++)', source_code)
    
    # Update the hitable list size calculation
    source_code = re.sub(r'new hitable_list\(d_list, \d+\*\d+\+1\)', f'new hitable_list(d_list, {max_a*2}*{max_b*2}+1)', source_code)
    
    with open('main.cu', 'w') as f:
        f.write(source_code)

def run_ray_tracing(width, height, samples):
    """
    Run the ray tracing program with specified parameters and extract execution time.
    
    Args:
        width (int): Image width
        height (int): Image height
        samples (int): Number of samples per pixel
    
    Returns:
        float: Execution time in seconds
    """
    command = f"./ray_tracing {width} {height} {samples}"
    
    # Run the command and capture output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Extract execution time using regex
    time_match = re.search(r'took (\d+\.\d+) seconds', result.stderr)
    if time_match:
        return float(time_match.group(1))
    else:
        raise ValueError("Could not extract execution time from output")

def benchmark_object_count():
    """
    Benchmark ray tracing performance across different object counts.
    
    Returns:
        tuple: Lists of object counts and corresponding execution times
    """
    # Compile the program first
    compile_cuda_program()
    
    # Image and rendering parameters (kept constant)
    width = 1280
    height = 720
    samples = 10
    
    # Object count configurations (representing the loop limits)
    object_counts = [
        (1, 1),    # Small scene
        (3, 3),    # Small scene
        (5, 5),    # Small scene
        (7, 7),    # Medium scene
        (9, 9),    # Medium scene
        (11, 11),  # Original scene
    ]
    
    # Lists to store results
    counts = []
    times = []
    
    # Run benchmarks
    for max_a, max_b in object_counts:
        # Modify source file to change object count
        modify_cuda_source_object_count(max_a, max_b)
        
        # Recompile with new object configuration
        compile_cuda_program()
        
        # Run and time the program
        execution_time = run_ray_tracing(width, height, samples)
        
        # Calculate total object count
        total_objects = (max_a*2 + 1) * (max_b*2 + 1)
        
        # Store results
        counts.append(total_objects)
        times.append(execution_time)
        
        print(f"Object count {total_objects}: {execution_time:.4f} seconds")
    
    return counts, times

def plot_object_count_results(counts, times):
    """
    Create a performance plot of object counts vs execution times.
    
    Args:
        counts (list): List of object counts
        times (list): Corresponding execution times
    """
    plt.figure(figsize=(10, 6))
    plt.plot(counts, times, marker='o', linestyle='', label='Measured Data')
    plt.title('Execution Time vs. Number of Objects')
    plt.xlabel('Number of Objects')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    
    # Annotate each point with the actual count
    for i, (count, time) in enumerate(zip(counts, times)):
        plt.annotate(f'{count}', 
                     (count, time), 
                     xytext=(10, 10), 
                     textcoords='offset points')
    
    # Optional: Add a trend line
    z = np.polyfit(counts, times, 1)
    p = np.poly1d(z)
    plt.plot(counts, p(counts), "r--", label='Best Fit Line')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ray_tracing_object_count_performance.png')
    plt.close()

def main():
    counts, times = benchmark_object_count()
    plot_object_count_results(counts, times)

if __name__ == "__main__":
    main()