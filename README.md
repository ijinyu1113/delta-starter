# delta-starter

A practice run for NCSA Delta: submit one small GPU training job, watch it in the queue, and get the results back into this repo.

The job trains a small network to compute `(a + b) mod 97`. It finishes in under a minute.

| File | What it is |
| --- | --- |
| `train.py` | The training script. Writes `results/metrics.jsonl` and `results/model.pt`. |
| `job.slurm` | The batch script that asks Slurm for one GPU and runs `train.py`. |
| `logs/` | Where Slurm writes the job's printed output. |

## 0. Fork this repo (once)

On https://github.com/ijinyu1113/delta-starter, click **Fork**. That gives you your own copy at `github.com/<your-username>/delta-starter`, which you can push to.

Then clone your fork to your laptop:

```bash
git clone https://github.com/<your-username>/delta-starter.git
```

## 1. Log in to Delta

```bash
ssh <ncsa-username>@login.delta.ncsa.illinois.edu
```

It asks for your NCSA password, then a Duo prompt. SSH keys do not work for logging in to Delta.

You land on a **login node**. Login nodes are shared by everyone and are for editing files and submitting jobs. Do not train on them.

## 2. Clone the repo on Delta

```bash
cd ~
git clone https://github.com/<your-username>/delta-starter.git
cd delta-starter
```

## 3. Build the Python environment (once)

```bash
bash setup_env.sh
```

This takes a minute or two. It creates a folder `.venv/` holding the extra libraries listed in `requirements.txt` (right now, `transformers`). PyTorch itself comes from Delta's `pytorch-conda` module, so it is not installed again.

You do this on the login node because installing needs the internet and only has to happen once. The job script then switches the environment on with `source .venv/bin/activate` before running your code.

To add a library later: add a line to `requirements.txt` and run `bash setup_env.sh` again.

## 4. Find your account name

```bash
accounts
```

Look in the "Project" column for the name ending in `-gpu` (something like `abcd-delta-gpu`). That is the allocation your job is charged to.

## 5. Submit the job

```bash
sbatch --account=<your-gpu-account> job.slurm
```

Slurm prints `Submitted batch job 1234567`. That number is your job ID.

## 6. Watch the queue

```bash
squeue -u $USER
```

The `ST` column is the state: `PD` means waiting for a node, `R` means running. When the job no longer appears, it has finished.

Other commands you will use:

| Command | What it does |
| --- | --- |
| `scancel <jobid>` | Cancel a job. |
| `tail -f logs/slurm-<jobid>.out` | Follow the job's output while it runs (Ctrl-C to stop following). |
| `sacct -j <jobid>` | Show how a finished job ended (`COMPLETED`, `FAILED`, `TIMEOUT`). |

## 7. Check the output

```bash
cat logs/slurm-<jobid>.out
tail -3 results/metrics.jsonl
```

The log should say `device: cuda` and name the GPU. The last lines of `metrics.jsonl` should show `train_acc` and `test_acc` at or near 1.0.

If something went wrong, the error is in `logs/slurm-<jobid>.err`.

## 8. Get the results back

Commit the metrics on Delta and push:

```bash
git add results/metrics.jsonl
git commit -m "First Delta run"
git push
```

GitHub asks for your username and a **personal access token** in place of a password. Create one at GitHub → Settings → Developer settings → Personal access tokens, with access to this repo.

Then on your laptop:

```bash
git pull
```

`results/metrics.jsonl` is now on your laptop. The model weights (`model.pt`) and the Slurm logs are left out of git on purpose (see `.gitignore`); large files and logs do not belong in a repo. To copy one of those, run this from your laptop:

```bash
scp <ncsa-username>@login.delta.ncsa.illinois.edu:~/delta-starter/results/model.pt .
```

## Things to try next

1. Open `results/metrics.jsonl` and compare `train_acc` with `test_acc` epoch by epoch. Training accuracy reaches 1.0 well before test accuracy moves. Why might that be?
2. Rerun with `--weight-decay 0`. Edit the `srun` line in `job.slurm`, commit, push from your laptop, `git pull` on Delta, and resubmit. What changes?
3. Change `--time` in `job.slurm` to `00:00:10` and see what a timeout looks like in `sacct`.

## Notes on `job.slurm`

- `--partition=gpuA40x4` is the A40 GPU queue. It is charged at half the rate of the A100 queue and is plenty for small jobs.
- `--time=00:15:00` is the limit after which Slurm kills the job. Short limits get scheduled sooner.
- `module load pytorch-conda` provides Python with PyTorch. If the log says `device: cpu`, tell your mentor.
- Run `sbatch` from inside the `delta-starter` folder. The job starts in whatever folder you submitted from, and it expects `train.py` and `logs/` to be there.
