<div align="center">

# IWINAC 2026

Intrinsic motivation methods for goal exploration and skill improvement success in a simple robotic environment.

</div>

## Videos
Experiment videos can be found in folder experiment_videos.

## Quick Start

```bash
mkdir iwinac2026
cd iwinac2026
git clone https://github.com/mullerjaku/iwinac2026.git
# --- Create environment (choose ONE of the following) ---

# Conda env
conda create -n iwinac python=3.10.12 -y
conda activate iwinac

# Install dependiencies
pip install --upgrade pip
pip install -e .  # uses setup.py
```

## Running the Main Script

The primary entry point appears to be `motivations_run.py` (adjust if you use a different orchestrator):
```bash
python motivations_run.py
```

If you need to run a sub-goal tree experiment:
```bash
python sub_goal_tree.py
```

---
Maintained by Jakub Müller.
