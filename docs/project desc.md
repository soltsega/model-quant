Build Your Own Model Quantizer

Pick a small open-source model and define a real-world constraint, for example “run on a laptop with 4 GB of RAM” or “reduce model size by at least 60%”.

Your Task
Build from scratch in PyTorch. Write your own quantization code in raw PyTorch, without pre-built quantization libraries such as bitsandbytes or torch.ao.quantization, etc.

Test at least two configurations, for example per-tensor vs. per-channel, or 8-bit vs. 4-bit/2-bit.

Pick a winner. Choose the configuration that best meets your constraint and justify it with measurements.

Note: There is no single correct answer. The goal is to reason about engineering trade-offs and build a working solution, not just reproduce the slides.

Deliverables (submit in the group)

1. Public GitHub repo URL with a notebook or script containing your quantizer and evaluation code.

2. Hugging Face model URL for your quantized model.

3. Short report (1-2 pages, PDF) covering your constraint, the configurations you tried, what failed, your results table, and your final decision with reasons. 
Results table comparing all configurations by model size, quantization error or accuracy change, and inference speed if possible. Include it in both your report

Deadline: Wednesday, October 21, before 5:00 PM.