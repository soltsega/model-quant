# DistilBERT quantization comparison

The Jupyter notebook compares seven representations of a Hugging Face DistilBERT sentiment classifier on SST-2:

1. FP32 baseline
2. FP16
3. BF16
4. Symmetric INT8, per-tensor weights
5. Symmetric INT8, per-output-channel weights
6. Symmetric INT8, per-group weights (64 values per group)
7. Asymmetric INT8, per-tensor weights

The notebook includes model/data loading, raw-PyTorch INT8 quantizers, activation calibration, evaluation, visualizations, winner selection, and checkpoint saving. It does not use a pre-built quantization library. The per-group variant applies a separate symmetric scale to each block of 64 weights within each linear output row. The two existing symmetric INT8 paths are reused, so rerunning the notebook overwrites them instead of creating duplicates; the per-group scheme gets its own checkpoint path.

## How the comparison developed

I started with three configurations: FP32, symmetric INT8 per-tensor, and symmetric INT8 per-channel. Per-tensor INT8 made a slightly smaller file, but lost 1.95 percentage points of accuracy and missed the project's 1 pp tolerance. Per-channel INT8 added about 0.34 MB and reduced the accuracy loss to 0.39 pp. That was a useful trade-off to see in the first experiment, and it led me to test FP16/BF16, per-group scaling, and asymmetric INT8 as well. FP8 and INT4 were considered but are not implemented in this notebook.

The final saved run selected FP16 under the size-and-accuracy rule: it matched FP32 accuracy and reduced serialized size by about 50%. On this CPU, though, FP16 took 497 ms/example versus 47 ms/example for FP32. The result is a smaller model with preserved accuracy on the evaluation subset, not faster CPU inference. The INT8 variants also use floating-point operations after dequantization, so their timings should not be read as native INT8 performance.

Development also surfaced a few practical snags: Windows initially failed to initialize PyTorch's `c10.dll`, a missing `nn` import interrupted activation calibration, and Hugging Face data loading needed a working connection or a complete local cache. These were setup and notebook-execution issues rather than quantization results. The [summary report](report.md) has the full method, results, and limitations.

## Run

Use Python 3.10+ and install the dependencies:

```powershell
python -m pip install -r requirements.txt
jupyter notebook quantization_project.ipynb
```

Run all cells. The first run downloads the model and GLUE SST-2 into `artifacts/hf-cache`; later runs reuse the cache. The defaults are a seeded 512-example validation subset, 128 training examples for calibration, batch size 16, and maximum sequence length 128. Change the constants near the top of the notebook to run a larger evaluation.

The notebook saves successful FP16, BF16, and four INT8 checkpoints under `artifacts/`, plus `results.csv` and `results.json`. It reuses the two existing symmetric INT8 filenames. FP32 remains the cached source model and is not saved as a duplicate checkpoint. Export an HTML presentation with:

```powershell
jupyter nbconvert --to html quantization_project.ipynb
```

## Upload the saved FP16 model

The standalone upload script uses `artifacts/distilbert-sst2-fp16.pt` and the model/cache files produced by the completed notebook run. It does **not** rerun evaluation or quantization. You can prepare the standard Transformers files locally without logging in:

```powershell
python scripts/upload_fp16_to_hub.py --package-only
```

Review `artifacts/hf-fp16/`. When you are ready to upload, authenticate in a terminal with `hf auth login`, then run:

```powershell
python scripts/upload_fp16_to_hub.py
```

The repository is public by default; pass `--private` to create it as private. Use `--repo-id USERNAME/REPO_NAME` to select a different repository. The script checks that the logged-in account matches the repository namespace and checks that the FP16 artifact and its saved metrics exist before uploading.

## Measurement and limitations

The comparison reports checkpoint size, SST-2 accuracy, macro-F1, accuracy change from FP32, logit mean absolute error, confusion matrices, and latency. The project constraint is at most a 1 percentage point accuracy loss; among eligible variants, the notebook selects the smallest checkpoint and uses logit MAE as a tie-breaker.

INT8 weights and activations are quantized and dequantized in a custom fake-quant wrapper before ordinary floating-point `F.linear` operations. This measures numerical changes and stored checkpoint size, not native integer acceleration or runtime activation-memory savings. FP16/BF16 CPU support and speed vary by processor and PyTorch kernels; unsupported formats are reported without silent fallback. Results use a validation subset and depend on the machine and software versions.

