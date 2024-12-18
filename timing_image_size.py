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
    # Modify the program to accept command-line arguments for width, height, and samples
    # You might need to modify your C++ source to do this
    command = f"./ray_tracing {width} {height} {samples}"
    
    # Run the command and capture output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Extract execution time using regex
    time_match = re.search(r'took (\d+\.\d+) seconds', result.stderr)
    if time_match:
        return float(time_match.group(1))
    else:
        raise ValueError("Could not extract execution time from output")

def benchmark_ray_tracing():
    """
    Benchmark ray tracing performance across different image sizes.
    
    Returns:
        tuple: Lists of image widths and corresponding execution times
    """
    # Compile the program first
    compile_cuda_program()
    
    # Image sizes to test (width x height)
    image_sizes = [
        (200, 113),   # 16:9 aspect ratio
        (400, 225),
        (800, 450),
        (1280, 720),  # 720p
        (1920, 1080), # 1080p
        (2560, 1440), # 1440p
        (3840, 2160)  # 4K
    ]
    
    # Constant samples per pixel
    samples = 10
    
    # Lists to store results
    widths = []
    times = []
    
    # Run benchmarks
    for width, height in image_sizes:
        execution_time = run_ray_tracing(width, height, samples)
        widths.append(width * height)
        times.append(execution_time)
        print(f"Image size {width}x{height}: {execution_time:.4f} seconds")
    
    return widths, times

def plot_benchmark_results(widths, times):
    """
    Create a performance plot of image sizes vs execution times.
    
    Args:
        widths (list): List of image widths
        times (list): Corresponding execution times
    """
    coefficients = np.polyfit(widths, times, 1)
    best_fit_line = np.poly1d(coefficients)

    plt.figure(figsize=(10, 6))
    plt.plot(widths, times, marker='o', linestyle='', label='Measured Data')
    plt.plot(widths, best_fit_line(widths), 
             color='red', linestyle=':', 
             label=f'Best Fit Line')
    plt.title('Execution Time vs. Total Number of Pixels')
    plt.xlabel('Total Number of Pixels')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('ray_tracing_performance.png')
    plt.close()

def main():
    widths, times = benchmark_ray_tracing()
    plot_benchmark_results(widths, times)

if __name__ == "__main__":
    main()