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
- Levenberg-Marquardt training.
- MATLAB-style `mapminmax` scaling of both inputs and each target.

Table 4 uses MSE for P and N, but MATLAB `msereg` (MSE plus weight/bias
regularization) for A. The paper does not report the regularization ratio, so
the script uses MSE for all three and does not pretend that this undocumented
part can be reproduced exactly.

The original notebook omitted target scaling. This leaves LM fitting values in
the range 5-76 directly and is the main reason its training and validation errors
remain high.

Exact reproduction of Table 6 is not guaranteed. The paper does not publish the
trained weights, random initialization, preprocessing state, or stopping history.
This matters because each network has 181-291 parameters but only 16 training
observations. The script therefore runs deterministic restarts, selects a run using
only samples 23-28, and evaluates samples 17-22 once at the end.
