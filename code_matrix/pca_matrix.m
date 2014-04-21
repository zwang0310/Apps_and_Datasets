% construct mnist.mat
img1 = loadMNISTImages('train-images.idx3-ubyte'); % 784 x 60,000
img2 = loadMNISTImages('t10k-images.idx3-ubyte');  % 784 x 10,000
data = [img1 img2]; % uniq_chk = unique(data','rows');
clear img1 img2
% centering
data = data - mean(data,2) * ones(1, size(data,2)); 
save mnist.mat data


% construct gisette.mat
data_tr = load('gisette_train.data'); % 6,500 x 5,000
data_t = load('gisette_test.data');   % 6,000 x 5,000
data_v = load('gisette_valid.data');  % 1,000 x 5,000
data = [data_t; data_tr; data_v]';    % 5,000 x 13,500   % uniq_chk = unique(data','rows');
clear data_t data_tr data_v
% centering
data = data - mean(data,2) * ones(1, size(data,2));
save gisette.mat data


% construct epsilon.mat
addpath('/users/zwang/research/libsvm-3.1/matlab');
[data_label, data] = libsvmread('epsilon_normalized'); % data: 400,000 x 2,000 and sparse
data = full(data)'; % 2,000 x 400,000
% centering
data = data - mean(data,2) * ones(1, size(data,2));
save epsilon.mat data

% construct siam_compete2007.mat
% This is a multi-labeled dataset, need to use read_sparse_ml.c from
% http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/

addpath('/users/zwang/research/libsvm-3.1/matlab');
mex -largeArrayDims read_sparse_ml.c
[data_label, data, map] = read_sparse_ml('tmc2007_train.svm');
data = full(data)';
% centering
data = data - mean(data,2) * ones(1, size(data,2));
save siam_compete2007.mat data




