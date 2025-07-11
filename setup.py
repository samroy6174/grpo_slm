
---

### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="grpo_slm",
    version="0.1.0",
    description="Group Relative Policy Optimization for Small Reasoning LMs",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "gym>=0.24.0",
        "numpy>=1.23.0",
        "matplotlib>=3.7.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
    ],
    python_requires=">=3.8",
)
