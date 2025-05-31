#!/usr/bin/env python3
import subprocess
import sys
from collections import Counter

def run_test(n_runs=10):
    """Run simple_signature.py multiple times and check for deterministic output"""
    outputs = []
    
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}...", file=sys.stderr)
        try:
            result = subprocess.run([sys.executable, "simple_signature.py"], 
                                  capture_output=True, text=True, timeout=120)
            # Extract just the model response line
            for line in result.stdout.split('\n'):
                if line.startswith("Model response:"):
                    outputs.append(line)
                    break
        except subprocess.TimeoutExpired:
            print(f"Timeout on run {i+1}", file=sys.stderr)
            continue
    
    # Count unique outputs
    counter = Counter(outputs)
    print(f"\nResults from {len(outputs)} successful runs:")
    for output, count in counter.items():
        print(f"{count}x: {output}")
    
    if len(counter) == 1:
        print("\n✓ Output is deterministic!")
    else:
        print(f"\n✗ Found {len(counter)} different outputs - not deterministic")

if __name__ == "__main__":
    run_test(5)