# DistilBERT Quantization: Summary Report

**Date:** 21 October 2026  
**Task:** SST-2 binary sentiment classification  
**Evaluation device:** CPU (`torch 2.13.0+cpu`)

## Executive summary

This experiment compared seven representations of a DistilBERT model already fine-tuned on SST-2. The goal was to reduce serialized checkpoint size while keeping accuracy within 1 percentage point (pp) of the FP32 baseline. The notebook completed all seven configurations and saved results and six reduced-precision checkpoints.

FP16 was selected by the notebook's decision rule. Its checkpoint was 49.99% smaller than the FP32 reference payload and it matched FP32 accuracy on the 512-example evaluation subset. This is a size and measured-accuracy result, not a CPU speed improvement: FP16 latency was 497.2 ms/example compared with 46.9 ms/example for FP32. Of the INT8 variants, symmetric per-channel had the best accuracy and lowest measured latency, but its serialized size was slightly larger than FP16/BF16.

## How the experiment grew

I began with three configurations: the FP32 reference, symmetric INT8 with one scale per weight matrix (per-tensor), and symmetric INT8 with a separate scale for each output channel. That smaller first comparison made the main trade-off concrete. In the final run, per-tensor INT8 was a little smaller (138.73 MB versus 139.07 MB), but its accuracy fell 1.95 pp and missed the project's 1 pp tolerance. Per-channel INT8 cost about 0.34 MB more, but its accuracy loss was only 0.39 pp. The extra scales were a small storage cost for a useful accuracy recovery on this model.

That result gave me a reason to broaden the experiment instead of adding formats just to make the table longer. I added FP16 and BF16 to see how ordinary lower-precision floating-point casts compared with integer quantization, then added symmetric groupwise INT8 and asymmetric per-tensor INT8 to test two more design choices: finer weight ranges and a nonzero zero point. The final notebook therefore compares seven configurations. FP8 and INT4 were considered but are not part of this implementation; I kept the finished comparison to methods implemented and evaluated in this notebook on the available CPU setup.

One finding I did not expect was that smaller files did not mean faster CPU inference. FP16 and BF16 cut the serialized size roughly in half, but took about 10.6 and 11.5 times as long per example as FP32 in the saved run. That changed how I interpret the winner: FP16 is the choice under the project's size-and-accuracy rule, not a recommendation for faster inference on this laptop.

## Objective and method

The model was [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), evaluated on a seeded subset of the GLUE SST-2 validation split. The model card describes this checkpoint as DistilBERT fine-tuned on SST-2. No model training or fine-tuning was performed in this project.

The notebook used seed 42, 128 SST-2 training examples to calibrate activation ranges, and 512 validation examples for evaluation. Inputs were truncated to 128 tokens and evaluated in batches of 16. The FP32 model provided the reference logits and metrics. The compared configurations were FP32, FP16, BF16, symmetric INT8 per-tensor, symmetric INT8 per-output-channel, symmetric INT8 per-group (64 weights per group), and asymmetric INT8 per-tensor.

FP16 and BF16 were produced by casting a fresh copy of the full model. The four INT8 variants were implemented in raw PyTorch. They fake-quantize the weights and inputs to linear layers, then dequantize them before ordinary floating-point linear operations. Activation ranges were measured per linear layer from the calibration subset. The groupwise scheme uses one weight scale for each 64 consecutive weights along an output row. The notebook reports serialized payload size, accuracy, macro-F1, accuracy change from FP32, mean absolute logit error (MAE), per-example forward latency, and confusion matrices.

The scope was kept deliberately practical for the machine available: CPU only, fixed seeded subsets, and a self-contained notebook rather than a separate quantization runtime. That makes the comparison repeatable and inspectable, but it also means these timings describe this PyTorch CPU path rather than optimized INT8 or GPU inference.

## Results

All seven configurations completed. Results below are from the saved notebook run and `artifacts/results.csv`.

| Configuration | Serialized size (MB) | Size reduction | Accuracy | Δ accuracy | Macro-F1 | Logit MAE | CPU latency (ms/example) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 267.85 | — | 90.82% | 0.000 pp | 0.9080 | 0.00000 | 46.9 |
| FP16 | 133.95 | 49.99% | 90.82% | 0.000 pp | 0.9080 | 0.00125 | 497.2 |
| BF16 | 133.95 | 49.99% | 90.63% | −0.195 pp | 0.9061 | 0.00947 | 539.7 |
| Symmetric INT8, per-tensor | 138.73 | 48.21% | 88.87% | −1.953 pp | 0.8883 | 0.49932 | 90.4 |
| Symmetric INT8, per-channel | 139.07 | 48.08% | 90.43% | −0.391 pp | 0.9038 | 0.43844 | 79.0 |
| Symmetric INT8, per-group (64) | 144.12 | 46.20% | 89.65% | −1.172 pp | 0.8962 | 0.42708 | 99.7 |
| Asymmetric INT8, per-tensor | 138.74 | 48.20% | 90.04% | −0.781 pp | 0.9002 | 0.18847 | 90.0 |

The accuracy rule admitted FP16, BF16, symmetric INT8 per-channel, and asymmetric INT8 per-tensor. FP16 and BF16 had the smallest serialized size, tied at the reported precision; FP16 had lower logit MAE, so it won the stated tie-break rule. The symmetric per-tensor and per-group INT8 variants exceeded the allowed accuracy loss.

Among INT8 variants, asymmetric per-tensor had the lowest logit MAE. Symmetric per-channel had the highest accuracy and lowest measured latency. Groupwise INT8 had the largest INT8 artifact and missed the accuracy tolerance, so the additional granularity did not improve this run's size/accuracy trade-off.

## Interpretation and limitations

- FP16/BF16 checkpoints were about half the FP32 serialized size. FP16 matched baseline accuracy on this subset, while BF16 missed by one example (0.195 pp). This does not establish that FP16 is generally more accurate; the comparison used one 512-example sample.
- Every reduced-precision representation was slower than FP32 on this CPU in this run. FP16 and BF16 were especially slow. The INT8 code uses floating-point kernels after dequantization, so its timings are not native integer inference benchmarks.
- INT8 serialized files were 46.2–48.2% smaller than FP32, rather than a theoretical 75% reduction for fully INT8 weights. The implementation quantizes `nn.Linear` weights and inputs; embeddings, layer normalization, biases, and other non-linear parameters remain floating point, and quantization scales add metadata.
- The reported values are specific to this CPU, PyTorch build, sample, and benchmark. A single run does not characterize timing variability or full-validation generalization. The measured latency should be repeated before making a deployment decision.
- The FP32 reference size is a serialized state payload measurement; the notebook does not save a duplicate FP32 `.pt` file. The six generated reduced-precision `.pt` files are present in `artifacts/`. INT8 files contain custom wrapper state dictionaries and are not drop-in Hugging Face checkpoints.

## Challenges encountered

The work involved more than choosing bit widths. On Windows, PyTorch initially failed to initialize its `c10.dll` (`WinError 1114`), which stopped the notebook before the quantization code could run. Once execution progressed, a missing `torch.nn` alias caused a `NameError` during activation calibration; putting the required imports in the setup cell fixed that notebook-level issue. At another point, loading the SST-2 validation split failed because the computer could not connect to Hugging Face. That traceback came from the network request, rather than from the quantizer, and the dataset had to be available from the Hub or its cache before the run could continue.

I also had to make sure the notebook's displayed results really came from the latest saved run. Jupyter reported that the file on disk was newer than the open session, and some earlier outputs were stale. I checked the final table against both `results.csv` and `results.json`, and refreshed the HTML export after the notebook was saved. For the final metrics in this report, I used that latest saved run rather than the earlier measurements.

The numerical side required choices too. Activation quantization needs a scale and zero point, so the notebook calibrates ranges on training examples and reserves validation examples for evaluation. The simple min/max method is easy to explain and reproduce, but outlier activations can stretch its range and reduce the precision available to typical values. I kept this as a clear baseline and recorded it as a limitation rather than implying it is the only or best calibration method.

## Conclusion

For the stated objective—approximately halving serialized size while staying within 1 pp of FP32 accuracy—FP16 is the selected configuration for this run. For CPU inference speed, these results do not support choosing a reduced-precision variant over FP32. Symmetric per-channel INT8 is the strongest INT8 result on accuracy and measured latency, while asymmetric per-tensor INT8 is the strongest INT8 result on logit MAE. The experiment demonstrates the trade-off among representation size, output fidelity, and device-specific latency; it does not demonstrate a general-purpose speedup.

## References

1. Hugging Face, [DistilBERT fine-tuned on SST-2 model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
2. Hugging Face, [GLUE dataset card](https://huggingface.co/datasets/nyu-mll/glue), SST-2 configuration.
3. LocalLLM.in, [The Complete Guide to LLM Quantization](https://localllm.in/blog/quantization-explained). Used for background on precision, calibration, and size/quality/speed trade-offs; its illustrative estimates are not reported as project measurements.
