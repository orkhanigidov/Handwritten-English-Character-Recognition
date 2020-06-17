function [train, valid, test] = loadEMNIST()
    % Load the training data
    X = loadImages('dataset/emnist-letters-train-images-idx3-ubyte');
    y = loadLabels('dataset/emnist-letters-train-labels-idx1-ubyte')';
    
    % Reshape X, flip and 90 degrees counterclockwise rotate
    X = flip(rot90(reshape(X, 28, 28, 1, length(y))));
    
    % Split training data into training set and validation set
    train.X = X(:, :, :, 1:88800);
    train.y = categorical(y(1:88800)');
    
    valid.X = X(:, :, :, 88801:end);
    valid.y = categorical(y(88801:end)');
    
    % Load the testing data
    X = loadImages('dataset/emnist-letters-test-images-idx3-ubyte');
    y = loadLabels('dataset/emnist-letters-test-labels-idx1-ubyte')';
    
    % Reshape X, flip and 90 degrees counterclockwise rotate
    X = flip(rot90(reshape(X, 28, 28, 1, length(y))));
    
    % Place these in the testing set
    test.X = X;
    test.y = categorical(y');
end