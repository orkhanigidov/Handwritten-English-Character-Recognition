[file, path] = uigetfile('*.png');
if isequal(file, 0)
    disp('User selected Cancel');
else
    disp(['User selected ', fullfile(path, file)]);
end

load matlab.mat;                          % load dataset from file into workspace
prev_x = 0;
prev_y = 0;

I = imread(fullfile(path, file));         % read image
I = imcomplement(I);                      % complement image
I = rgb2gray(I);                          % convert RGB image to grayscale
BW = imbinarize(I);                       % binarize 2-D grayscale image
[~, n] = bwlabel(BW);                     % label connected components in 2-D binary image
stats = regionprops(BW, 'BoundingBox');   % measure properties of image regions

figure                                    % create figure window
imshow(I);                                % display image

for i = 1:n
    idx = stats(i).BoundingBox;           % get bounding boxes for all the regions
    
    x = idx(1);
    y = idx(2);
    w = idx(3);                           % height
    h = idx(4);                           % width
    pos = [x y w h];                      % [x y width height]
    
    rectangle("Position", pos, ...
              "EdgeColor", "r", ...
              "LineWidth", 2);            % plot bounding boxes
    
    J = imcrop(BW, pos);                  % crop image
    J = imresize(J, [28 28]);             % resize image
%     J = imgaussfilt(double(J), 2);        % 2-D Gaussian filtering of images
    
    label = classify(net, J);             % classify data using a trained deep learning neural network
    
    if x - prev_x > 200                   % finding blank spaces more than 200 pixels
        fprintf(" ");                     % print space between words
    end
    fprintf(char(64 + double(label)));    % print detected text
    
    pause(1);                             % stop MATLAB execution 1 second
    
    prev_x = x;
    prev_y = y;
end
fprintf("\n");                            % newline