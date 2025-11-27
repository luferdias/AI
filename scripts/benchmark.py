#!/usr/bin/env python3
"""
Benchmark script for model performance evaluation.
Measures latency, throughput, and resource usage.
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List
import logging

import torch
import numpy as np
import psutil
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.models.deeplab import DeepLabV3Plus
from src.data.preprocessor import ImagePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark model performance."""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize benchmark.
        
        Args:
            model: PyTorch model to benchmark
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.preprocessor = ImagePreprocessor(target_size=(512, 512))
    
    def warmup(self, num_iterations: int = 10):
        """Warmup model with dummy data."""
        logger.info(f"Warming up model ({num_iterations} iterations)...")
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        
        logger.info("Warmup complete")
    
    def measure_latency(self, num_iterations: int = 100) -> dict:
        """
        Measure inference latency.
        
        Args:
            num_iterations: Number of iterations
            
        Returns:
            Dictionary with latency statistics
        """
        logger.info(f"Measuring latency ({num_iterations} iterations)...")
        
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        latencies = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="Latency test"):
                start = time.time()
                _ = self.model(dummy_input)
                
                # Synchronize for accurate GPU timing
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)
        
        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        }
    
    def measure_throughput(self, batch_sizes: List[int] = [1, 4, 8, 16]) -> dict:
        """
        Measure throughput for different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with throughput results
        """
        logger.info("Measuring throughput...")
        
        results = {}
        num_iterations = 50
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size {batch_size}...")
            
            dummy_input = torch.randn(batch_size, 3, 512, 512).to(self.device)
            total_samples = 0
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.model(dummy_input)
                    total_samples += batch_size
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            throughput = total_samples / elapsed_time
            
            results[f'batch_{batch_size}'] = {
                'throughput_samples_per_sec': throughput,
                'time_per_sample_ms': (elapsed_time / total_samples) * 1000
            }
        
        return results
    
    def measure_memory(self) -> dict:
        """
        Measure memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        logger.info("Measuring memory usage...")
        
        # System memory
        process = psutil.Process()
        mem_info = process.memory_info()
        
        results = {
            'system_memory_mb': mem_info.rss / 1024 / 1024,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            ) / 1024 / 1024
        }
        
        # GPU memory
        if self.device == 'cuda':
            results['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            results['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return results
    
    def run_full_benchmark(self) -> dict:
        """
        Run complete benchmark suite.
        
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("=" * 60)
        logger.info("Starting Full Benchmark")
        logger.info("=" * 60)
        
        # Warmup
        self.warmup()
        
        # Latency
        latency_results = self.measure_latency()
        
        # Throughput
        throughput_results = self.measure_throughput()
        
        # Memory
        memory_results = self.measure_memory()
        
        return {
            'latency': latency_results,
            'throughput': throughput_results,
            'memory': memory_results
        }
    
    def print_results(self, results: dict):
        """Print benchmark results in formatted way."""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        
        # Latency
        logger.info("\nLatency Statistics:")
        logger.info("-" * 60)
        for key, value in results['latency'].items():
            logger.info(f"  {key:20s}: {value:8.2f} ms")
        
        # Check target latency
        if results['latency']['p95_ms'] < 500:
            logger.info("  ✓ Target latency (<500ms) achieved!")
        else:
            logger.warning("  ✗ Target latency (<500ms) not achieved")
        
        # Throughput
        logger.info("\nThroughput:")
        logger.info("-" * 60)
        for batch, metrics in results['throughput'].items():
            logger.info(f"  {batch}:")
            logger.info(f"    Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
            logger.info(f"    Time/sample: {metrics['time_per_sample_ms']:.2f} ms")
        
        # Memory
        logger.info("\nMemory Usage:")
        logger.info("-" * 60)
        for key, value in results['memory'].items():
            if 'parameters' in key:
                logger.info(f"  {key:30s}: {value:,}")
            else:
                logger.info(f"  {key:30s}: {value:.2f}")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument(
        '--model',
        type=str,
        choices=['unet', 'deeplab'],
        default='unet',
        help='Model to benchmark'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations for latency test'
    )
    
    args = parser.parse_args()
    
    # Initialize model
    logger.info(f"Initializing {args.model} model...")
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    else:
        model = DeepLabV3Plus(n_channels=3, n_classes=1, pretrained=False)
    
    # Run benchmark
    benchmark = ModelBenchmark(model, device=args.device)
    results = benchmark.run_full_benchmark()
    benchmark.print_results(results)


if __name__ == '__main__':
    main()
