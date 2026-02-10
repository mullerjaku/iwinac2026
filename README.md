<div align="center">

# IWINAC 2026

Intrinsic motivation methods for goal exploration and skill improvement success in a simple robotic environment.

</div>

## Videos
Experiment videos can be found in folder experiment_videos.

## Quick Start

```bash
git clone https://github.com/mullerjaku/project_motivations.git
cd project_motivations
# --- Create environment (choose ONE of the following) ---

# Option A: Built-in venv (recommended if you don't use Conda)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Option B: Conda
conda create -n motivations python=3.11 -y
conda activate motivations

# Install dependiencies
pip install --upgrade pip
pip install -e .  # uses setup.py
```

## Running the Main Script

The primary entry point appears to be `motivations_run.py` (adjust if you use a different orchestrator):
```bash
python3 motivations_run.py
```

If you need to run a sub-goal tree experiment:
```bash
python3 sub_goal_tree.py
```

## Project Structure (excerpt)
```
classes/        # Motivation-related modules (curiosity, effectance, entropy, prediction, etc.)
env/            # Environment / robot simulation components
data/           # JSON / text inputs (goals, traces, utility values)
motivations_run.py
sub_goal_tree.py
setup.py
```

### Updating the User Task
The file `user_task.txt` (at repository root) defines the current high-level user instruction for the system (e.g. goal semantics or desired behavior).

To change it, simply overwrite its content with a single concise sentence, for example:
```
Pick up only green objects and ignore red ones.
```
After editing, rerun `python3 motivations_run.py` so components that parse the task can adapt.

## Contributing
Pull requests and issues welcome. Please keep changes modular and document new classes.

## License
Specify a license (e.g. MIT) – currently not declared.

## Roadmap / TODO
- [ ] Add automated tests
- [ ] Provide minimal dataset example generator
- [ ] Add benchmarking script for curiosity vs. random exploration
- [ ] Document each motivation module in a separate markdown file
- [ ] Add license file

---
Maintained by Jakub Müller.
