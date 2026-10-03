# FCC Feed ANN Modeling

Python reproduction of the eight-input Model 2 in Dasila et al. (2014).

Run:

```bash
python -m pip install -r requirements.txt
python fcc_ann_reproduction.py FCC_feed_data.xlsx --starts 10
```

The paper uses the following setup:

- Inputs: `Sp. Gr`, `5 %`, `10 %`, `30 %`, `50 %`, `70 %`, `90 %`, and `95 %`.
- Samples 1-16: training; samples 23-28: testing/model selection; samples 17-22: final validation.
- Three independent networks, followed by P/N/A normalization to 100%.
- P architecture: 12 and 13 hidden neurons.
- N architecture: 12 and 12 hidden neurons.
- A architecture: 9 and 9 hidden neurons.
- Hidden activations: `tansig` then `logsig`; output activation: `purelin`.
- Pure-PyTorch Levenberg-Marquardt training with MATLAB's documented defaults:
  `mu=0.001`, `mu_dec=0.1`, `mu_inc=10`, `mu_max=1e10`, `min_grad=1e-7`,
  and six validation failures.
- MATLAB-style `mapminmax` scaling of both inputs and each target.
- Deterministic Nguyen-Widrow initialization for every seeded restart.

Table 4 uses MSE for P and N, but MATLAB `msereg` for A. The script implements
`ratio*MSE + (1-ratio)*MSW`, including the parameter residuals in the LM
Jacobian. It defaults to MATLAB's `ratio=0.5`; use `--msereg-ratio` to test a
different value if the authors used a non-default setting.

The original notebook omitted target scaling. This leaves LM fitting values in
the range 5-76 directly and is the main reason its training and validation errors
remain high.

Exact reproduction of Table 6 is not guaranteed. The paper does not publish the
trained weights, random initialization, preprocessing state, or stopping history.
This matters because each network has 181-291 parameters but only 16 training
observations. The script therefore runs deterministic restarts, selects a run using
only samples 23-28, and evaluates samples 17-22 once at the end.

This is a behavioral reimplementation, not a guarantee of identical MATLAB
floating-point trajectories. PyTorch and the unknown MATLAB release can still
differ in Nguyen-Widrow details, linear-system solvers, and accepted LM steps.

## MATLAB Online

Upload `FCC_feed_data.xlsx` and `fcc_ann_matlab.m` to the same folder in
[MATLAB Online](https://matlab.mathworks.com/), open `fcc_ann_matlab.m`, and
click **Run**. Deep Learning Toolbox is required. The script writes
`fcc_matlab_predictions.csv` and saves the trained networks as
`fcc_matlab_models.mat`.

The paper is ambiguous about whether samples 23-28 were used for early stopping
or only post-training testing. The script defaults to test-only behavior, which
matches the paper's terminology. Run the alternative by changing
`useTestingRowsForEarlyStopping` to `true`. Keep samples 17-22 untouched in both
cases because they are the published final validation rows.
