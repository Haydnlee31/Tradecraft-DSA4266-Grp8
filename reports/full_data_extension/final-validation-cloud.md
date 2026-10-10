# Completed validation replay and recovery instructions

This replay has now completed successfully for all 27 candidates. Keep these
instructions for recovery; do not repeat the job merely because the historical
handoff remains here. The current next stage is in the
[test preparation handoff](test-preparation.md).

Use the existing T4 machine and its `tradecraft-official39` environment. Do not
create a larger instance, upgrade Torch, retrain, or open test shards for this
check. These commands are the original handoff; the completed run is recorded
in the [CUDA receipt](final-validation-cuda-checks.json).

## 1. Start the existing machine when ready, then connect from your Mac

```bash
ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o IdentitiesOnly=yes -i "/Users/haydn/Downloads/trade_ssh_01.pem" ubuntu@haydn-tradecraft-02.nus.cloud
```

Confirm the prompt is `ubuntu@...`, not your Mac. On the cloud terminal:

```bash
cd /home/ubuntu/Tradecraft-DSA4266-Grp8
git branch --show-current
git status --short
```

Expect `experiment/full-data-scaling`. Stop if the branch differs or tracked
files have modifications. An untracked virtual environment alone is not a
tracked-file modification. Fetch via the explicit GitHub URL so this also works
if an old `origin` still points to a transferred Git bundle:

```bash
git fetch https://github.com/Haydnlee31/Tradecraft-DSA4266-Grp8.git experiment/full-data-scaling && git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

If authentication or networking fails, stop and request a bundle handoff; do
not reset the worktree or paste a token into chat.

## 2. Check the existing environment and inputs

```bash
source /home/ubuntu/.venvs/tradecraft-official39/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pip check
python -m unittest tests.test_official_final_evaluation tests.test_official_test_panel -q
python -c "import torch; print(torch.__version__); assert torch.__version__ == '2.7.0+cu128'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(3, device='cuda') * 2).cpu().tolist())"
ls -lh /home/ubuntu/tradecraft-official39-ablation-backup.tgz /home/ubuntu/tradecraft-official39-ablation-confirmation-backup.tgz
ls outputs/official39-packed-2m-v1/manifest.json outputs/official39-ablation-inputs-v1/official39-ablation-panel-v1/receipt.json
```

Only continue when these checks succeed. The 21 unit tests use synthetic data;
they do not open the official test split. The runner rechecks archive, data,
panel and checkpoint hashes itself. The validation arrays already reside in
the packed data; no test upload is required.

If either backup archive is missing on the cloud machine, use a **second Mac
terminal** to copy the reviewed backups, then return to the cloud terminal:

```bash
scp -o ConnectTimeout=20 -o ServerAliveInterval=30 -o IdentitiesOnly=yes -i "/Users/haydn/Downloads/trade_ssh_01.pem" "/Users/haydn/Downloads/tradecraft-official39-ablation-backup.tgz" "/Users/haydn/Downloads/tradecraft-official39-ablation-confirmation-backup.tgz" ubuntu@haydn-tradecraft-02.nus.cloud:/home/ubuntu/
```

## 3. Replay the fixed validation endpoints

Use an existing tmux session if already inside one. Otherwise, on the cloud:

```bash
tmux new-session -A -s tradecraft-final-validation
```

After attaching, this one-line command uses an explicit interpreter, directory
and CUDA configuration, so it does not depend on the tmux shell's activation:

```bash
cd /home/ubuntu/Tradecraft-DSA4266-Grp8 && CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/ubuntu/.venvs/tradecraft-official39/bin/python -u scripts/official39_final_evaluation.py validation-replay --archives /home/ubuntu --packed outputs/official39-packed-2m-v1 --validation-panel outputs/official39-ablation-inputs-v1/official39-ablation-panel-v1 --device cuda --output outputs/official39-final-validation-replay-cuda-v1
```

It prints one line after each candidate's two validation passes. There is no
training loop or official test evaluation. Do not infer it is stuck from a
quiet first archive/data check. Exact cloud duration has not been measured.

For a completed run, inspect the concise gate summary:

```bash
/home/ubuntu/.venvs/tradecraft-official39/bin/python -c "import json; p='/home/ubuntu/Tradecraft-DSA4266-Grp8/outputs/official39-final-validation-replay-cuda-v1/checks.json'; r=json.load(open(p)); print({k:r[k] for k in ('status','candidates_verified','exact_same_runtime_references','exact_historical_endpoints','test_scoring_ready','test_opened','test_evaluated')})"
```

The target is `status=complete`, both exact counts `27`,
`test_scoring_ready=True`, and both test flags `False`. **Stop and share this
summary even if it passes.** It does not authorize test preparation or scoring.
If any count differs, keep the artifacts; do not loosen tolerances or rerun
training. `blocked_replay_mismatch` is an intentional blocked gate, not success.

An existing output directory is intentionally protected. After a normal
interruption, rerun the same replay command with `--resume` appended; completed
case hashes are checked and inference is skipped for those cases. If it reports
a running lock or an orphan case, request inspection. Do not delete the output
or lock simply because SSH disconnected. You can detach tmux with Ctrl-B, then
D; reconnect using the SSH command in step 1 and reattach the same session.

Keep the resulting folder until its receipt and backup are reviewed. No cloud
training or final-test command is included in this handoff.
