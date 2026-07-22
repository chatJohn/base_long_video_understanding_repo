# Base Long Video Understanding Repo

This repository is a development workspace for keyframe selection and LVLM-input keyframe construction in long-video understanding.

The current goal is to build, modify, and evaluate a two-stage workflow on top of an existing WFS-SB codebase. First, we select keyframes from long videos and generate `keyframe_indices`. Then, when those selected frames are passed into an LVLM, we construct the actual visual input used by the model.  
## Development Purpose

This repo is maintained for our own keyframe selection and LVLM-input construction development, including:

- understanding and refactoring the existing WFS pipeline;
- developing new keyframe selection strategies;
- comparing wavelet-based, relevance-based, diversity-based, and hybrid sampling methods;
- generating `keyframe_indices` files for long-video benchmarks;
- constructing the final LVLM visual input from selected keyframes;
- integrating constructed keyframe inputs into LVLM evaluation workflows;
- keeping experiment code, configs, and benchmark adapters in one place.


## Reference

This repository is based on the original WFS-SB project:

- Original repository: https://github.com/MAC-AutoML/WFS-SB
- Paper: https://arxiv.org/abs/2603.00512

If using or comparing against the original method, please refer to the original WFS-SB paper and repository for citation and reproduction details.
