import os
import sys

sys.path.insert(0, os.path.abspath("."))

with open("src/agents/hrl_12worker_dispatcher.py", "r") as f:
    orig = f.read()

with open(".scratch/test_compounding_flywheel.py", "r") as f:
    test = f.read()

print("orig length:", len(orig), "test length:", len(test))
