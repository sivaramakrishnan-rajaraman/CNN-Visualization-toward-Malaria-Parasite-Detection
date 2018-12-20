%%This code evaluates the custom model for a given dataset and classifies
%%them into normal and abnormal classes. In the process we also visualize
%%the learned features and activations in the trained model.
%%
% you can give the path to your training and testing data. If you are performing 
%5-fold cross-validation, the training and testing data from
%the respective folds need to be given

train_folder = 'f1_mal/train/';
test_folder = 'f1_mal/test/';
categories = {'abnormal', 'normal'};

% |imageDatastore| recursively scans the directory tree containing the
% images. Folder names are automatically used as labels for each image.

trainImages = imageDatastore(fullfile(train_folder, categories), 'LabelSource', 'foldernames'); 
testImages = imageDatastore(fullfile(test_folder, categories), 'LabelSource', 'foldernames');
trainLabels = trainImages.Labels;
testLabels = testImages.Labels;
abnormal = find(trainImages.Labels == 'abnormal', 1);
normal = find(trainImages.Labels == 'normal', 1);
subplot(2,1,1);
imshow(readimage(trainImages,abnormal))
subplot(2,1,2);
imshow(readimage(trainImages,normal))
%% architecture of the custom model
layers = [ ...
    imageInputLayer([100 100 3],'Normalization', 'zerocenter')
    convolution2dLayer(3,32,'Stride',1)
    %batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Stride',1)
    %batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,128,'Stride',1)
    %batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,256,'Stride',1)
    %batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,512,'Stride',1)
    %batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(512)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer 
    classificationLayer()];
    
% declare training options
opts = trainingOptions('sgdm','Verbose',true,...
      'LearnRateSchedule','none',...
      'L2Regularization',3.6e-06,...
      'MaxEpochs',300,... 
      'MiniBatchSize',32,...
      'ValidationData',{testImages,testLabels},...
      'ValidationFrequency',30,...
      'ValidationPatience',10,...
      'Plots','training-progress',...
      'Momentum',0.99,...
      'InitialLearnRate',1.9935e-04,...
      'ExecutionEnvironment','gpu'); 
convnet = trainNetwork(trainImages,layers,opts); 

%% Feature Visualization
% Let us visualize the learned weights in the convolutional layers of the
% trained model

%print the layers in the trained model
convnet.Layers 

%To produce images that resemble each class the most closely, 
%select the final fully connected layer, and set channels to be the indices of the classes.

% Let us visualize the convolutional layers. The convolutional layers are 2, 
% 5, 8, 11, 14, and fully connected layers are 16 and 18 in this model. 

%% Let us visualize the first convolutional layer 
%Visualize the feature maps learned by this layer using deepDreamImage by 
% setting channels to be the vector of indices 1:32. 
% Set 'PyramidLevels' to scale the images. To display the images together, 
%you can use montage (Image Processing Toolbox). 
% deepDreamImage uses a compatible GPU, by default, if available. 
%Otherwise it uses the CPU. A CUDA-enabled NVIDIA GPU with compute 
% capability 3.0 or higher is required for training on a GPU.

layer = 2; 
name = convnet.Layers(layer).Name
channels = 1:32;
I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',2, 'ExecutionEnvironment','gpu',...
    'PyramidScale',1.4,'NumIterations',100); 
figure
montage(I)
title(['Layer ',name,' Features'])

% Next layer conv2
channels = 1:32;
layer = 5; 
name = convnet.Layers(layer).Name
I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',2, 'ExecutionEnvironment','gpu',...
    'PyramidScale',1.4,'NumIterations',100); 
figure
montage(I)
title(['Layer ',name,' Features'])

% Next layer conv3
layer = 8; 
name = convnet.Layers(layer).Name
channels = 1:32;
I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',2, 'ExecutionEnvironment','gpu',...
    'PyramidScale',1.4,'NumIterations',100); 
figure
montage(I)
title(['Layer ',name,' Features'])

% Next layer conv4
layer = 11; 
name = convnet.Layers(layer).Name
channels = 1:32;
I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',2, 'ExecutionEnvironment','gpu',...
    'PyramidScale',1.4,'NumIterations',100); 
figure
montage(I)
title(['Layer ',name,' Features'])


% Next layer conv5
layer = 14; 
name = convnet.Layers(layer).Name
channels = 1:32;
I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',2, 'ExecutionEnvironment','gpu',...
    'PyramidScale',1.4,'NumIterations',100); 
figure
montage(I)
title(['Layer ',name,' Features'])
%% Visualizing fully connected layers
%For the fully connected layer at layer 16, we will visualize the first two
% features maps learned. To suppress detailed output on the optimization process, 
%set 'Verbose' to 'false' in the call to deepDreamImage. Notice that the layers
% which are deeper into the network yield more detailed filters.

layer = 16; 
name = convnet.Layers(layer).Name;
channels = 1:2;
for layer = layer
    I = deepDreamImage(convnet,layer,channels, ...
        'PyramidLevels',2, 'ExecutionEnvironment','gpu',...
        'PyramidScale',1.4,'Verbose',true, 'NumIterations',100);
    figure
    montage(I)
    title(['Layer ',name,' Features'])
end

%To produce images that resemble each class the most closely, 
% select the final fully connected layer and set channels to be the indices
% of the classes. The images generated from the final fully connected layer
% correspond to the image classes.

layer = 18; 
name = convnet.Layers(layer).Name;
channels = [1 2];
convnet.Layers(end).ClassNames(channels)
I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',2, 'ExecutionEnvironment','gpu',...
    'PyramidScale',1.4,'Verbose',true, 'NumIterations',100);
figure
montage(I)
title(['Layer ',name,' Features'])

%% Visualizing the activations in the learned model
% let us visualize the intermediate activations in the trained model. We
% will input an image and visualize how the different filters in the
% convolutional layers respond to different patterns in the input image.
% Let us input an abnormal cell and learn the activations. 

im = imread('C182P143NThinF_IMG_20151201_172216_cell_151.png'); 
imshow(im)
imgSize = size(im);
imgSize = imgSize(1:2);

% Investigate activations in the first convolutional layer by 
% observing which areas in the convolutional layers activate on an image
% and comparing with the corresponding areas in the original images. 
% Each layer of a convolutional neural network consists of many 2-D arrays called channels. 
% Pass the image through the network and examine the output activations 
% of the conv1 layer.

act1 = activations(convnet,im,'conv_1','OutputAs','channels','ExecutionEnvironment','gpu');

%The activations are returned as a 3-D array, with the third dimension 
% indexing the channel on the conv1 layer. To show these activations using 
% the montage function, reshape the array to 4-D. The third dimension in 
% the input to montage represents the image color. 
%Set the third dimension to have size 1 because the activations do not have color. 
%The fourth dimension indexes the channel.

sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);

%Now we can visualize the activations. Each activation can take any value, 
% so normalize the output using mat2gray. All activations are scaled so 
% that the minimum activation is 0 and the maximum is 1. 
%Display a montage of the 32 images on an 4-by-8 grid, one for each channel
% in the layer.

figure 
montage(mat2gray(act1),'Size',[4 8])

%Each square in the montage of activations is the output of a 
% channel in the conv1 layer. White pixels represent strong positive 
% activations and black pixels represent strong negative activations. 
% A channel that is mostly gray does not activate as strongly on the input 
% image. The position of a pixel in the activation of a channel 
% corresponds to the same position in the original image. 
%A white pixel at some location in a channel indicates that the channel 
% is strongly activated at that position.
 
%Take a particular channel, say channel 24. This shows the highest 
% activitation. Resize the activations in channel 24 to have the same size
% as the original image and display the activations.

act1ch24 = act1(:,:,:,24);
act1ch24 = mat2gray(act1ch24);
act1ch24 = imresize(act1ch24,imgSize);
figure
imshowpair(im,act1ch24,'montage')
%You can see that this channel activates on the location of the parasite, 
%because the whiter pixels in the channel correspond to location of the 
% parasite. You also can try to find interesting channels by programmatically 
%investigating channels with large activations. Find the channel with the 
% largest activation using the max function, resize, and show the activations.
 
[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);
figure
imshowpair(im,act1chMax,'montage')
%% Investigate a Deeper Layer
%Most convolutional neural networks learn to detect features 
% like color and edges in their first convolutional layer. 
%In deeper convolutional layers, the network learns to detect more 
% complicated features. Later layers build up their features by combining 
% features of earlier layers. Investigate the conv3 layer in the same way 
% as the conv1 layer. Calculate, reshape, and show the activations in a montage.

act3 = activations(convnet,im,'conv_3','OutputAs','channels'); 
sz = size(act3);
act3 = reshape(act3,[sz(1) sz(2) 1 sz(3)]);
figure
montage(imresize(mat2gray(act3),[22 22]))

%There are too many images to investigate in detail, 
% so focus on some of the more interesting ones. Display the strongest activation 
% in the conv3 layer.

[maxValue3,maxValueIndex3] = max(max(max(act3)));
act3chMax = act3(:,:,:,maxValueIndex3);
imshow(imresize(mat2gray(act3chMax),imgSize))

%In this case, the maximum activation channel is interesting
% for detailed features and shows strong activation on the parasitic location.
montage(imresize(mat2gray(act3(:,:,:,[79])),imgSize))

%Many of the channels contain areas of activation that are both light and dark. 
%These are positive and negative activations, respectively. 
%However, only the positive activations are used because of the rectified 
% linear unit (ReLU) that follows the conv_3 layer.
%To investigate only positive activations, repeat the analysis to 
% visualize the activations of the relu3 layer.
 
act3relu = activations(convnet,im,'relu_3','OutputAs','channels');
sz = size(act3relu);
act3relu = reshape(act3relu,[sz(1) sz(2) 1 sz(3)]);
figure
montage(imresize(mat2gray(act3relu(:,:,:,[79])),imgSize))

%Compared to the activations of the conv3 layer, the activations of the 
% relu3 layer clearly pinpoint areas of the parasite. 
%% testing the model for activations
%Test Whether a Channel Recognizes parasite
%Check whether channels 79 of the relu3 layer activate on parasites. 
%Input a new normal image without parasite to the network and compare the 
% resulting activations with the activations of the original image.
%Read and show the image with no parasite and compute the activations 
% of the relu3 layer.

imClosed = imread('C65P26N_ThinF_IMG_20150818_154436_cell_173.png');
act3Closed = activations(convnet,imClosed,'relu_3','OutputAs','channels');
sz = size(act3Closed);
act3Closed = reshape(act3Closed,[sz(1),sz(2),1,sz(3)]);

%Plot the images and activations in one figure.

channelsClosed = repmat(imresize(mat2gray(act3Closed(:,:,:,[79])),imgSize),[1 1 3]);
figure
imshow(channelsClosed)
imsave
channelsOpen = repmat(imresize(mat2gray(act3relu(:,:,:,[79])),imgSize),[1 1 3]);
montage(cat(4,im,channelsOpen*255,imClosed,channelsClosed*255));
title('Input Image, Channel 79');

%% Heatmap Computation
% We will compute the probability map and superimpose 
% it on the original image. This gives a heatmap kind of 
% visualization that focusses on the prominent ROI in the image
% used by the model to arrive at the decisions. 

img = imread('C65P26N_ThinF_IMG_20150818_154436_cell_173.png');
act = imread('cell_181.png');
img = imresize(img, [100 100]);
act = imresize(act, [100 100]);
I = im2double(img(:,:,1));
probabilityMap = im2double(act(:,:,1));
outputImage = f_myOverlay(I,probabilityMap,0,0.5);
imshow(outputImage);
imsave
%% 