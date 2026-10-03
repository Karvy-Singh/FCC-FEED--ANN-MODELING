%% FCC feed ANN reproduction of Dasila et al. (2014), Model 2
% Upload this file and FCC_feed_data.xlsx to the same MATLAB Online folder.
% This script requires Deep Learning Toolbox.

clear;
clc;

workbook = "FCC_feed_data.xlsx";
numberOfStarts = 10;
mseregRatio = 0.5;

% The paper does not state whether its six "testing" samples controlled
% early stopping. true matches the Python reproduction. Set false to train
% without validation stopping and use samples 23-28 only for model selection.
useTestingRowsForEarlyStopping = true;

raw = readcell(workbook, "Sheet", "Feed Data");
data = cell2mat(raw(3:30, 1:17));

sampleNumber = data(:, 1);
featureColumns = [2, 7, 8, 9, 10, 11, 12, 13];
targetColumns = [15, 16, 17];

% MATLAB neural networks use variables-by-samples orientation.
X = data(:, featureColumns)';
T = data(:, targetColumns)';

trainIndices = 1:16;
validationIndices = 17:22; % Final untouched rows reported in Table 6.
testingIndices = 23:28;

[netP, infoP] = trainTargetNetwork( ...
    X, T(1, :), [12, 13], 'mse', 0, numberOfStarts, ...
    trainIndices, testingIndices, useTestingRowsForEarlyStopping, mseregRatio);
[netN, infoN] = trainTargetNetwork( ...
    X, T(2, :), [12, 12], 'mse', 1000, numberOfStarts, ...
    trainIndices, testingIndices, useTestingRowsForEarlyStopping, mseregRatio);
[netA, infoA] = trainTargetNetwork( ...
    X, T(3, :), [9, 9], 'msereg', 2000, numberOfStarts, ...
    trainIndices, testingIndices, useTestingRowsForEarlyStopping, mseregRatio);

fprintf("P: seed=%d, best_epoch=%d, test_RMSE=%.3f\n", ...
    infoP.seed, infoP.bestEpoch, infoP.testRMSE);
fprintf("N: seed=%d, best_epoch=%d, test_RMSE=%.3f\n", ...
    infoN.seed, infoN.bestEpoch, infoN.testRMSE);
fprintf("A: seed=%d, best_epoch=%d, test_RMSE=%.3f\n", ...
    infoA.seed, infoA.bestEpoch, infoA.testRMSE);

XValidation = X(:, validationIndices);
rawPrediction = [netP(XValidation); netN(XValidation); netA(XValidation)];
prediction = 100 .* rawPrediction ./ sum(rawPrediction, 1);
observed = T(:, validationIndices);

errors = observed(:) - prediction(:);
rmse = sqrt(mean(errors .^ 2));
sse = sum(errors .^ 2);
sst = sum((observed(:) - mean(observed(:))) .^ 2);
rSquared = 1 - sse / sst;

results = table( ...
    sampleNumber(validationIndices), ...
    observed(1, :)', prediction(1, :)', ...
    observed(2, :)', prediction(2, :)', ...
    observed(3, :)', prediction(3, :)', ...
    'VariableNames', {'Sample', 'P_exp', 'P_pred', 'N_exp', 'N_pred', ...
    'A_exp', 'A_pred'});

disp("Held-out validation (samples 17-22):");
disp(results);
fprintf("RMSE = %.3f\n", rmse);
fprintf("R2   = %.4f\n", rSquared);
fprintf("Paper Table 6 Model 2: RMSE = 1.71, R2 = 0.994\n");

writetable(results, "fcc_matlab_predictions.csv");
save("fcc_matlab_models.mat", "netP", "netN", "netA", ...
    "infoP", "infoN", "infoA", "rmse", "rSquared");

%% Local functions
function [bestNet, bestInfo] = trainTargetNetwork( ...
    X, target, architecture, performanceFunction, seedOffset, numberOfStarts, ...
    trainIndices, testingIndices, useEarlyStopping, mseregRatio)

    bestTestRMSE = inf;
    bestNet = [];
    bestInfo = struct();

    for restart = 0:(numberOfStarts - 1)
        seed = seedOffset + restart;
        rng(seed, "twister");

        net = feedforwardnet(architecture, 'trainlm');
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'logsig';
        net.layers{3}.transferFcn = 'purelin';

        net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
        net.outputs{3}.processFcns = {'removeconstantrows', 'mapminmax'};
        net.performFcn = performanceFunction;
        if strcmp(performanceFunction, 'msereg')
            net.performParam.ratio = mseregRatio;
        end

        net.divideFcn = 'divideind';
        net.divideParam.trainInd = trainIndices;
        if useEarlyStopping
            net.divideParam.valInd = testingIndices;
            net.divideParam.testInd = [];
        else
            net.divideParam.valInd = [];
            net.divideParam.testInd = testingIndices;
        end

        net.trainParam.epochs = 1000;
        net.trainParam.goal = 0;
        net.trainParam.min_grad = 1e-7;
        net.trainParam.max_fail = 6;
        net.trainParam.mu = 1e-3;
        net.trainParam.mu_dec = 0.1;
        net.trainParam.mu_inc = 10;
        net.trainParam.mu_max = 1e10;
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;

        % Configure processing ranges from training rows only, then force the
        % initialization method used by classic MATLAB feed-forward networks.
        net = configure(net, X(:, trainIndices), target(:, trainIndices));
        for layer = 1:numel(net.layers)
            net.layers{layer}.initFcn = 'initnw';
        end
        net = init(net);
        [net, trainingRecord] = train(net, X, target);

        testingPrediction = net(X(:, testingIndices));
        testRMSE = sqrt(mean((target(:, testingIndices) - testingPrediction) .^ 2));
        if testRMSE < bestTestRMSE
            bestTestRMSE = testRMSE;
            bestNet = net;
            bestInfo.seed = seed;
            bestInfo.bestEpoch = trainingRecord.best_epoch;
            bestInfo.testRMSE = testRMSE;
            bestInfo.trainingRecord = trainingRecord;
        end
    end
end
