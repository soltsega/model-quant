# Reducing DistilBERT for SST-2: a quantization study

I wanted to see what happens when a model is made smaller: how much accuracy do I give up, and does a smaller checkpoint actually run faster on my computer? I used a DistilBERT sentiment classifier and compared floating-point formats with several INT8 quantization choices. The experiment gave me a concrete answer for this model and my CPU, rather than a general claim that one format is always best.

## The model and task

The starting point was [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), a DistilBERT model already fine-tuned to classify movie reviews as positive or negative. I evaluated it on the SST-2 validation split from the [GLUE dataset](https://huggingface.co/datasets/nyu-mll/glue). I did not train or fine-tune the model; I focused on changing its numerical representation after training.

I used 128 SST-2 training sentences to calibrate activation ranges and a seeded subset of 512 validation examples for evaluation. Inputs were capped at 128 tokens. The FP32 model was my reference. The experiment compared FP16, BF16, symmetric INT8 per tensor, symmetric INT8 per output channel, symmetric INT8 per group of 64 weights, and asymmetric INT8 per tensor.

## How the experiment took shape

I began with three versions: the FP32 baseline, symmetric per-tensor INT8, and symmetric per-channel INT8. That first comparison showed me that the scale granularity mattered: per-tensor INT8 produced a slightly smaller checkpoint, but lost more accuracy. Per-channel INT8 used a little more space for its extra scales and preserved more of the baseline accuracy. I then added FP16 and BF16 to compare integer quantization with simpler floating-point casts, and added groupwise and asymmetric INT8 to test further choices.

The INT8 quantizers are implemented directly in PyTorch. For each linear layer, the notebook calibrates activation ranges from the training examples, calculates quantization scales and zero points, and quantizes weights and inputs. It then dequantizes them before using ordinary floating-point linear operations. FP16 and BF16 are produced by casting the model weights.

## Results from my CPU run

The notebook's rule was to allow at most a 1 percentage-point accuracy drop from FP32. Among the configurations that met that rule, it selected the smallest serialized checkpoint, using logit error to break ties.

| Configuration | Serialized size (MB) | Accuracy | Macro-F1 | Change in accuracy | Latency (ms/example) |
|---|---:|---:|---:|---:|---:|
| FP32 baseline | 267.85 | 90.82% | 0.9080 | 0.000 pp | 46.9 |
| FP16 | 133.95 | 90.82% | 0.9080 | 0.000 pp | 497.2 |
| BF16 | 133.95 | 90.63% | 0.9061 | −0.195 pp | 539.7 |
| Symmetric INT8, per tensor | 138.73 | 88.87% | 0.8883 | −1.953 pp | 90.4 |
| Symmetric INT8, per channel | 139.07 | 90.43% | 0.9038 | −0.391 pp | 79.0 |
| Symmetric INT8, per group (64) | 144.12 | 89.65% | 0.8962 | −1.172 pp | 99.7 |
| Asymmetric INT8, per tensor | 138.74 | 90.04% | 0.9002 | −0.781 pp | 90.0 |

**FP16 was the winner under that rule.** Its serialized payload was about half the size of FP32 and it matched the FP32 accuracy on these 512 examples. That result surprised me a little: FP16 was much slower on my CPU, at about 497 ms per example compared with 47 ms for FP32. So I can call it the size-and-accuracy winner in this run, but not the speed winner.

Among the INT8 variants, per-channel quantization had the highest accuracy and the lowest measured latency. Per-tensor symmetric INT8 missed the accuracy limit, while asymmetric per-tensor INT8 stayed within it. Groupwise quantization did not improve this particular size-and-accuracy trade-off. These outcomes are specific to this model, data subset, PyTorch build, and machine.

## What I learned and what the numbers do not show

The experiment helped me see that lower bit width alone does not determine the result. The number and placement of scales affect accuracy and storage, and the hardware and kernels determine whether lower precision helps with speed. The simplest configuration was not the most accurate, and the smallest eligible configuration did not run fastest on my CPU.

There is an important limit to the INT8 results: this is **fake quantization**, not a benchmark of optimized integer inference. Values are quantized and dequantized, then processed with floating-point operations. The reported serialized sizes are for the notebook's saved payloads; the INT8 files are custom PyTorch wrapper state dictionaries, not drop-in Hugging Face INT8 checkpoints. The latency figures therefore do not demonstrate native INT8 speedup or activation-memory savings.

I also had to work through a few practical issues before I could trust the comparison: PyTorch's DLL initialization failed on Windows at first, a missing `torch.nn` import interrupted calibration, and the notebook's outputs needed refreshing after edits so they matched the saved results. Those problems were separate from the quantization method, but solving them was part of getting a reproducible run.

The results are from one 512-example validation subset and one CPU run. A larger evaluation and repeated timing runs would give a stronger basis for deployment decisions. For the current project, the result I can defend is narrower: **FP16 gave the best size-and-accuracy trade-off under my stated rule, while FP32 remained faster on the CPU I tested.**

## Run the notebook

Install the requirements and open the notebook:

```powershell
python -m pip install -r requirements.txt
jupyter notebook notebooks/quantization_project.ipynb
```

Run all cells. The first run downloads the model, tokenizer, and SST-2 data; later runs can reuse the cache. The notebook writes its metrics to `artifacts/results.csv` and `artifacts/results.json`, saves successful reduced-precision checkpoints under `artifacts/`, and displays comparison plots. The rendered notebook is available at [`blog/quantization_project.html`](blog/quantization_project.html).

For the full technical write-up and detailed caveats, see [`docs/report.md`](docs/report.md).
