clear; clc

%% Step 1 Load dataset
[train, valid, test] = loadEMNIST();

%% Step 2 Specify the convolutional neural network architecture
imageSize = [28 28 1];

layers = [
    imageInputLayer(imageSize)                    % 28x28x1 images with 'zerocenter' normalization
    
    convolution2dLayer(5, 32, "Padding", "same")  % 32 5x5x1 convolutions with stride [1  1] and padding 'same'
    batchNormalizationLayer                       % Batch normalization with 32 channels
    reluLayer                                     % ReLU
    
    maxPooling2dLayer(2, "Stride", 2)             % 2x2 max pooling with stride [2  2] and padding 'same'
    dropoutLayer(0.2)                             % 20% dropout
    
    convolution2dLayer(3, 64, "Padding", "same")  % 64 3x3x32 convolutions with stride [1  1] and padding 'same'
    batchNormalizationLayer                       % Batch normalization with 64 channels
    reluLayer                                     % ReLU
    
    maxPooling2dLayer(2, "Stride", 2)             % 2x2 max pooling with stride [2  2] and padding 'same'
    dropoutLayer(0.2)                             % 20% dropout
    
    fullyConnectedLayer(128)                      % 128 fully connected layer
    batchNormalizationLayer                       % Batch normalization with 128 channels
    reluLayer                                     % ReLU
    dropoutLayer(0.2)                             % 20% dropout
    
    fullyConnectedLayer(26)                       % 26 fully connected layer
    softmaxLayer                                  % softmax
    classificationLayer];                         % crossentropyex

%% Step 3 Specify training options
options = trainingOptions("adam", ...             % set the solver for training network
    "ExecutionEnvironment", "parallel", ...       % turn on automatic parallel support
    "InitialLearnRate", 0.001, ...                % set the initial learning rate
    "MaxEpochs", 10, ...                          % set the MaxEpochs
    "MiniBatchSize", 200, ...                     % set the MiniBatchSize
    "Shuffle", "every-epoch", ...                 % set the data shuffling
    "ValidationData", {valid.X, valid.y}, ...     % set the Validation data
    "ValidationPatience", Inf, ...                % patience of validation stopping
    "Plots", "training-progress", ...             % turn on the training progress plot
    "Verbose", true);                             % send command line output

%% Step 4 Train the network
net = trainNetwork(train.X, train.y, layers, options);

%% Step 5 Predicting on test set
YPred = classify(net, test.X);
accuracy = sum(YPred == test.y) / numel(test.y);

%% Step 6 Save convolutional neural network model
filename = "matlab.mat";
save(filename);