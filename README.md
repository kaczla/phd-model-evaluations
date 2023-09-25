# Model evaluations

This repository contains script for training and evaluating Neural Language Models.

Training contains:
- fine-tuning MLM (encoder) models on GLUE Benchmark,
- fine-tuning CLM (decoder) models on GLUE Benchmark,
- fine-tuning Seq2Seq (encoder-decoder) models on GLUE Benchmark.

Evaluation contains:
- intrinsic evaluation (loss),
- LM-GAP evaluation (PerplexityHashed),
- extrinsic evaluation (downstream task: GLUE Benchmark, SuperGLUE Benchmark).
