# Running Challenges on the Jed HPC Cluster

This guide explains how to connect to the Jed cluster, set up your environment, and submit jobs for Challenges 1b and 1c.

---

## 1. Connect to the Cluster

Use your EPFL Gaspar account to SSH into Jed:

```bash
ssh -l <username> jed.hpc.epfl.ch
```

> Replace `<username>` with your EPFL Gaspar username.

---

## 2. One-Time Setup

Run these steps **once** after your first login to set up your project environment.

### Load required modules

```bash
module purge
module load gcc python py-virtualenv
```

### Clone your fork and set up the virtual environment

```bash
mkdir ER_course
cd ER_course
git clone <your_fork_url>
cd micro-515-EvoRob

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Replace `<your_fork_url>` with the URL of your forked repository.

---

## 3. Submitting a Job

Each time you log in and want to run a job, navigate to your project folder and activate your environment first:

```bash
cd ER_course/micro-515-EvoRob
source .venv/bin/activate
```

Then submit your SLURM script with:

```bash
sbatch run_challenge1a.sh
# or
sbatch run_challenge1b.sh
# or
sbatch run_challenge1c.sh
```

---

## 4. SLURM Scripts

These scripts are already in your `micro-515-EvoRob` project folder.

### `run_challenge1a.sh` — Morphology Optimisation

```bash
#!/bin/bash
#SBATCH --job-name=ER_Challenge1a
#SBATCH --output=logs/ER_Challenge1a_%j.out
#SBATCH --error=logs/ER_Challenge1a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source .venv/bin/activate
mkdir -p logs
python Challenge1a.py
```

### `run_challenge1b.sh` — Evolutionary Optimisation

```bash
#!/bin/bash
#SBATCH --job-name=ER_Challenge1b
#SBATCH --output=logs/ER_Challenge1b_%j.out
#SBATCH --error=logs/ER_Challenge1b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source .venv/bin/activate
mkdir -p logs
python Challenge1b.py
```

### `run_challenge1c.sh` — Reinforcement Learning (PPO)

```bash
#!/bin/bash
#SBATCH --job-name=ER_Challenge1c
#SBATCH --output=logs/ER_Challenge1c_%j.out
#SBATCH --error=logs/ER_Challenge1c_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source .venv/bin/activate
mkdir -p logs
python Challenge1c.py
```

---

## 5. Monitoring Your Job

Check the status of your submitted jobs:

```bash
squeue -u <username>
```

Cancel a job:

```bash
scancel <job_id>
```

View job output logs in real time:

```bash
tail -f logs/challenge1b_<job_id>.out
```

---

## 6. Tips

- **`--time`**: Adjust the time limit based on your training duration. Challenge 1b (100 generations, population 10) is relatively fast; Challenge 1c with 10M timesteps may take several hours.
- **Checkpoints**: Both scripts save results to `./results/`. Make sure this folder is writable.
- **Environment**: Always `source .venv/bin/activate` before running or submitting jobs to ensure the correct Python packages are used.
