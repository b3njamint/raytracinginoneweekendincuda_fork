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

def modify_cuda_source(tx, ty):
    """
    Modify the CUDA source file to change block sizes.
    
    Args:
        tx (int): X-dimension of thread block
        ty (int): Y-dimension of thread block
    """
    with open('main.cu', 'r') as f:
        source_code = f.read()
    
    # Replace block size definitions
    source_code = re.sub(r'int tx = \d+;', f'int tx = {tx};', source_code)
    source_code = re.sub(r'int ty = \d+;', f'int ty = {ty};', source_code)
    
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

def benchmark_block_sizes():
    """
    Benchmark ray tracing performance across different block sizes.
    
    Returns:
        tuple: Lists of block sizes and corresponding execution times
    """
    # Compile the program first
    compile_cuda_program()
    
    # Image and rendering parameters (kept constant)
    width = 1280
    height = 720
    samples = 10
    
    # Block sizes to test
    block_sizes = [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32)
    ]
    
    # Lists to store results
    block_dimensions = []
    times = []
    
    # Run benchmarks
    for tx, ty in block_sizes:
        # Modify source file to change block sizes
        modify_cuda_source(tx, ty)
        
        # Recompile with new block sizes
        compile_cuda_program()
        
        # Run and time the program
        execution_time = run_ray_tracing(width, height, samples)
        
        # Store results
        block_dimensions.append((tx, ty))
        times.append(execution_time)
        
        print(f"Block size {tx}x{ty}: {execution_time:.4f} seconds")
    
    return block_dimensions, times

def plot_block_size_results(block_dimensions, times):
    """
    Create a performance plot of block sizes vs execution times.
    
    Args:
        block_dimensions (list): List of (tx, ty) block dimensions
        times (list): Corresponding execution times
    """
    # Convert block dimensions to total threads per block
    total_threads = [tx for tx, ty in block_dimensions]
    
    plt.figure(figsize=(10, 6))
    plt.plot(total_threads, times, marker='o')
    plt.title('Execution Time vs. Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.xticks(total_threads, [f'{tx}x{ty}' for tx, ty in block_dimensions])
    
    # Annotate each point with actual block size
    for i, (threads, time) in enumerate(zip(total_threads, times)):
        plt.annotate(f'{block_dimensions[i][0]}x{block_dimensions[i][1]}', 
                     (threads, time), 
                     xytext=(10, 10), 
                     textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('ray_tracing_block_size_performance.png')
    plt.close()

def main():
    block_dimensions, times = benchmark_block_sizes()
    plot_block_size_results(block_dimensions, times)

if __name__ == "__main__":
    main()