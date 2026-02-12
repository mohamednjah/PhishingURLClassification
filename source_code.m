% Import the dataset from a CSV file into a table
data = readtable('PhiUSIIL_Phishing_URL_Dataset.csv');

% Group data by labels
labelCounts = groupcounts(data.label);

% Display the counts
disp(labelCounts);

% Specify the columns that are non-numeric (string features) and remove them
stringColumns = {'FILENAME', 'URL', 'Domain', 'TLD', 'Title'};
data(:, stringColumns) = [];

% Extract labels (assumes the label column is named 'label')
numericLabels = data.label;

% Remove the label column to create the features table
features = data;
features.label = [];
% Remove DomtainTitleMatch Score because it highly correlates to URLMatchScore
features.DomainTitleMatchScore = [];
numericFeatures = table2array(features);


% Compute the correlation matrix (using 'complete' to handle any missing data)
corrMatrix = corr(numericFeatures, 'Rows', 'complete');


% Create a heatmap for the correlation matrix
figure;
heatmap(features.Properties.VariableNames, features.Properties.VariableNames, corrMatrix);
title('Correlation Matrix of Numeric Features');


% -----------------------
% 1. Split the Data
% -----------------------
cv = cvpartition(size(numericFeatures,1), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

trainFeatures = numericFeatures(trainIdx, :);
testFeatures  = numericFeatures(testIdx, :);
trainLabels   = numericLabels(trainIdx, :);
testLabels    = numericLabels(testIdx, :);

% -----------------------
% 2. Normalize the Data
% -----------------------
[trainFeaturesNorm, mu, sigma] = zscore(trainFeatures);
testFeaturesNorm = (testFeatures - mu) ./ sigma;

%% -----------------------
% 3. Train the SOM
% -----------------------
gridSize = [3 3];   % Define a 3x3 SOM grid (9 neurons)
net = selforgmap(gridSize);
net = train(net, trainFeaturesNorm');

%% -----------------------
% 4. Assign Neuron Labels (Training Phase)
% -----------------------
% Get the BMU indices for each training sample
trainOutput = net(trainFeaturesNorm');
[~, trainBMU] = max(trainOutput, [], 1);
trainBMU = trainBMU';

% Determine the number of neurons (should be gridSize(1)*gridSize(2))
numNeurons = prod(gridSize);
neuronLabels = -ones(numNeurons, 1);  % initialize with -1 for neurons with no samples

% For each neuron, assign the label based on majority vote of training samples mapping to it
for i = 1:numNeurons
    idx = find(trainBMU == i);
    if ~isempty(idx)
        neuronLabels(i) = mode(trainLabels(idx));
    end
end

% -----------------------
% 5. Testing Phase: Predict Labels for Test Samples
% -----------------------
testOutput = net(testFeaturesNorm');
[~, testBMU] = max(testOutput, [], 1);
testBMU = testBMU';

% Assign predicted labels based on the neuron's assigned label
predictedTestLabels = zeros(size(testBMU));
for i = 1:length(testBMU)
    predictedTestLabels(i) = neuronLabels(testBMU(i));
end

% -----------------------
% 6. Model Evaluation
% -----------------------
% Compute classification accuracy
accuracy = sum(predictedTestLabels == testLabels) / length(testLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Create and display a confusion matrix
figure;
confMat = confusionmat(testLabels, predictedTestLabels);
confusionchart(confMat);
title('Confusion Matrix for SOM Classification');

% -----------------------
% 7. Additional Evaluation Metrics
% -----------------------

% 2. Plot the SOM hits - shows number of training samples mapped to each neuron
figure;
plotsomhits(net, trainFeaturesNorm');
title('SOM Hits for Training Data');

% 3. Plot the weight planes for each feature dimension
figure;
plotsomplanes(net);
title('SOM Weight Planes');

% Calculate precision, recall, F1 score, and specificity per class
nClasses = size(confMat, 1);
precision = zeros(nClasses, 1);
recall    = zeros(nClasses, 1);
f1score   = zeros(nClasses, 1);
specificity = zeros(nClasses, 1);
support   = zeros(nClasses, 1);

totalSamples = sum(confMat(:));
for i = 1:nClasses
    TP = confMat(i,i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    TN = totalSamples - TP - FP - FN;
    
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
    specificity(i) = TN / (TN + FP + eps);
    support(i) = sum(confMat(i, :));
end

% Display per-class metrics
for i = 1:nClasses
    fprintf('Class %d: Precision=%.2f%%, Recall=%.2f%%, F1 Score=%.2f%%, Specificity=%.2f%%, Support=%d\n', ...
        i, precision(i)*100, recall(i)*100, f1score(i)*100, specificity(i)*100, support(i));
end

% Compute macro-averaged metrics
macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1 = mean(f1score);
macroSpecificity = mean(specificity);
fprintf('Macro-Averaged Metrics: Precision=%.2f%%, Recall=%.2f%%, F1 Score=%.2f%%, Specificity=%.2f%%\n', ...
    macroPrecision*100, macroRecall*100, macroF1*100, macroSpecificity*100);