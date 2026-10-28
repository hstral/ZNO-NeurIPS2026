# Third-party notices

This repository contains original research code for the Z-Domain Neural
Operator (ZNO) and uses third-party projects, datasets, and benchmark assets.
The top-level `LICENSE` applies only to the original code and documentation.
Third-party materials remain under their own licenses.

- **Official S5 (JAX/Flax).** The S5 baseline uses the unmodified upstream
  implementation by Smith, Warrington, and Linderman, "Simplified State Space
  Layers for Sequence Modeling", ICLR 2023
  (<https://github.com/lindermanlab/S5>, commit
  `3c18fdb6b06414da35e77b94b9cd855f6a95ef17`, MIT License, Copyright (c) 2022
  Linderman Lab). It is not redistributed here.
  `scripts/fetch_official_s5.sh` clones it into `third_party/S5`. The code in
  `s5_baseline/` adapts it to sequence regression.
- **FNO baseline.** This follows the Fourier Neural Operator of Li et al.,
  "Fourier Neural Operator for Parametric Partial Differential Equations",
  ICLR 2021. It is implemented through the `neuraloperator` package (MIT
  License).
- **S4D baseline (`s4d_official`).** This is a local PyTorch implementation of
  the S4D-Lin parameterization of Gu et al., "On the Parameterization and
  Initialization of Diagonal State Space Models", NeurIPS 2022. It follows the
  minimal S4D layer of the `state-spaces/s4` repository
  (<https://github.com/state-spaces/s4>, Apache License 2.0).
- **LNO and the 1D ODE benchmark.** The 1D ODE benchmark data
  (`1D_*/Data/data.mat`) and the related reference code are adapted from the
  Laplace Neural Operator release by Cao, Goswami, and Karniadakis, "Laplace
  Neural Operator for Solving Differential Equations", Nature Machine
  Intelligence 2024 (MIT License; the notice is reproduced below).
- **SysIdentPy.** The AOLS/FROLS polynomial and Fourier NARMAX baselines use
  SysIdentPy (BSD 3-Clause License).
- **Public nonlinear system-identification benchmarks.** These are loaded
  through the `nonlinear-benchmarks` Python package. Dataset licenses and
  citation requirements remain those of the original benchmark providers
  (Silverbox, Wiener–Hammerstein, Cascaded Tanks, CED, EMPS).
- **CUDA/Triton.** The optional fused ZNO scan uses CUDA/C++ and Triton
  kernels. A pure PyTorch reference path is kept for portability.

## Laplace Neural Operator MIT license notice

The 1D ODE benchmark assets and related reference code are adapted from the
Laplace Neural Operator release.

MIT License

Copyright (c) 2023 qianyingcao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
