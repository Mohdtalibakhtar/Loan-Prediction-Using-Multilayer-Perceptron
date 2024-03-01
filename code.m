% Load the data
data = readtable('finalfile.csv');

% Normalize the features (excluding the target variable 'loan_status')
features = data(:, 1:end-1); % Exclude the target variable
labels = data.loan_status; % Target variable
features = normalize(features);

% Split the data into training and test sets
cv = cvpartition(size(data, 1), 'HoldOut', 0.2); % Hold out 20% of the data for testing
idx = cv.test;
X_train = features(~idx, :);
Y_train = labels(~idx);
X_test = features(idx, :);
Y_test = labels(idx);

% Assuming 'data' is your loaded table

% Normalize the features (excluding the target variable 'loan_status')
% Assuming normalization is needed; MATLAB's normalize function can directly handle tables for feature columns but returns a table
features = data(:, 1:end-1); % Exclude the target variable column
labels = data.loan_status; % This is already an array if accessed this way

% Normalize features
for i = 1:width(features)
    features.(i) = (features.(i) - mean(features.(i))) ./ std(features.(i));
end

% Split the data into training and test sets
cv = cvpartition(size(data, 1), 'HoldOut', 0.2); % Hold out 20% of the data for testing
idx = cv.test;

% Directly indexing the table for X; converting to array if model requires array input
X_train = features(~idx, :);
X_test = features(idx, :);

% Labels are already vectors, so no need for conversion
Y_train = labels(~idx);
Y_test = labels(idx);

% If your model requires array inputs, convert X_train and X_test to arrays
X_train = table2array(X_train);
X_test = table2array(X_test);

% No need to convert Y_train and Y_test as they should already be vectors if extracted with dot notation
seedValue = 42;
rng(seedValue);





hiddenSizeOptions = {[64], [64, 32], [64, 32, 16]};
learningRateOptions = [0.1, 0.01, 0.001];
Y_train = categorical(Y_train);
Y_test = categorical(Y_test);

% Assuming X_train, Y_train, X_test, and Y_test are defined
results = trainAndEvaluateMLP(X_train, Y_train, X_test, Y_test, hiddenSizeOptions, learningRateOptions);

% Display results
disp(results);


function layers = createMLPModel(inputSize, hiddenSizes, outputSize)
    layers = [
        featureInputLayer(inputSize, 'Name', 'input')
    ];
    
    for i = 1:length(hiddenSizes)
        layers = [
            layers
            fullyConnectedLayer(hiddenSizes(i), 'Name', ['fc' + string(i)])
            reluLayer('Name', ['relu' + string(i)])
        ];
    end

    % Add two more hardcoded hidden layers
    layers = [
        layers
        fullyConnectedLayer(50, 'Name', 'fc_additional1')
        reluLayer('Name', 'relu_additional1')
        fullyConnectedLayer(50, 'Name', 'fc_additional2')
        reluLayer('Name', 'relu_additional2')
    ];
    
    % Adjust the output layer to have 2 units for binary classification
    layers = [
        layers
        fullyConnectedLayer(2, 'Name', 'output_fc') % Change here for binary classification
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
end



function results = trainAndEvaluateMLP(X_train, Y_train, X_test, Y_test, hiddenSizeOptions, learningRateOptions)
     % Initialize the results table with proper variable types if not already done
    results = table('Size', [0 4], ...
                    'VariableTypes', {'cell', 'double', 'double', 'double'}, ...
                    'VariableNames', {'HiddenSizes', 'LearningRate', 'TrainingTime', 'Accuracy'});
    
    for i = 1:length(hiddenSizeOptions)
        for j = 1:length(learningRateOptions)
            % Define the network
            inputSize = size(X_train, 2);
            outputSize = 1; % Adjust based on your task
            hiddenSizes = [64, 32, 16, 1];
            layers = createMLPModel(inputSize, hiddenSizeOptions{i}, outputSize);
            
            % Set training options
            options = trainingOptions('adam', ...
                'InitialLearnRate', learningRateOptions(j), ...
                'MaxEpochs', 100, ...
                'Verbose', false, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', {X_test, Y_test}, ...
                'Plots', 'training-progress');
            
            % Train the network
            tic;
            net = trainNetwork(X_train, Y_train, layers, options);
            trainingTime = toc;
            
            % Evaluate the network
            Y_pred = classify(net, X_test);
            accuracy = sum(Y_pred == Y_test) / numel(Y_test);
            
            % Store results
            results = [results; {hiddenSizeOptions{i}, learningRateOptions(j), trainingTime, accuracy}];
        end
    end
    
    results.Properties.VariableNames = {'HiddenSizes', 'LearningRate', 'TrainingTime', 'Accuracy'};
end


