% ===========================================================================
% MAIN FUNCTION FILE FOR A 3 LAYER FEED FORWARD NEURAL NETWORK
% Author: Hugo Nordell
% Copyright 2016
% ===========================================================================

% ===========================================================================
% 1. IMPORT DATA & SETUP NN ENV
% ===========================================================================
clc; clear all; close all;

fflush(stdout)
fileName = input('Enter the filename for data to import (csv) in quotation marks "file": ');

python('setLabels.py', fileName);

% Read CSV data
data = csvread(strcat(fileName(1:length(fileName)-4),'_numerics.csv'));

% Randomize the data set by trainings (not time series)
data = data(randperm(size(data,1)),:);

% Assuming y is the last column of data
X_full = data(:,1:size(data,2)-1); y_full = data(:,end);

% Select in_sample and out_sample of training set
ratio = 0.8;
X_in = X_full(1:length(X_full)*ratio,:); y_in = y_full(1:length(y_full)*ratio);
X_out = X_full(length(X_full)*ratio+1:end,:); y_out = y_full(length(y_full)*ratio+1:end);

% ===========================================================================
% 2. VISUALIZE IMAGE DATA
% ===========================================================================

% Determine if visualize data
viz_imgs = yes_or_no('Visualize data? ');
if (viz_imgs)
	plotData(X_in, y_in)
end

% ===========================================================================
% 3. LOOK AT COV & CORR MATRICES
% ===========================================================================
X_in_corr = corr(X_in);
X_in_cov = cov(X_in);

% Find check for multicollinearity & high correlations
corr_barrier = 0.8;
[corr_rows, corr_cols] = find(tril(X_in_corr - eye(size(X_in_corr))) > corr_barrier);

fprintf('\nThe covariance matrix for the features in-sample is:\n');
disp(X_in_cov)
fprintf('\nThe correlation matrix for the features in-sample is:\n');
disp(X_in_corr)

if (~isempty(corr_rows) || ~isempty(corr_cols))
	fprintf('\nThe following features exhibit pearson correlation above %f:\n', corr_barrier);
	for i = 1:length(corr_rows)
		fprintf('Features: %d and %d\n', corr_rows(i), corr_cols(i));
	end
	fprintf('\n');
end

% ===========================================================================
% 4. SET INPUT AND HIDDEN LAYER UNITS, PENALTIES AND FIND PARAMETERS FOR NN
% ===========================================================================
input_layer_size = length(X_in(1,:));
hidden_layer_size = 2*input_layer_size;
% Python script finds number of unique labels;
num_labels = csvread(strcat(fileName(1:length(fileName)-4),'_num_labels.csv'));

weight_options = { Bias = 'true'};
% random_theta_1 = randomWeights(input_layer_size, hidden_layer_size);
% random_theta_2 = randomWeights(hidden_layer_size, num_labels);

penalties = 0.1:0.3:10; % Check several penalties
MaxIter = 200; % Number of iterations before terminating min(cost)

% Find optimal parameters
[theta_1, theta_2, J_min, optimal_penalty] = optimalParams(X_in, y_in, input_layer_size, hidden_layer_size, num_labels, penalties, MaxIter);

viz_J = yes_or_no('Visualize cost function J over iterations? ');
if (viz_J)
	figure();
	plot([1:MaxIter], J_min);
	hold on;
	xlabel('Number of iterations');
	ylabel('Computed cost J');
	plot([1 MaxIter], [J_min(end) J_min(end)], 'r');
end

% ===========================================================================
% 5. PREDICT OUTCOMES, IN- & OUT-OF-SAMPLE & ACCURACY
% ===========================================================================
predictions_in = predictor(theta_1, theta_2, X_in);
accuracy_in = mean(predictions_in == y_in);
predictions_out = predictor(theta_1, theta_2, X_out);
accuracy_out = mean(predictions_out == y_out);

fprintf('\nAccuracy in-sample: %f\n', accuracy_in*100);
fprintf('\nAccuracy out-of-sample: %f\n', accuracy_out*100);

% Print some predictions out of sample
y_mat_out = [predictions_out y_out];
y_mat_out = y_mat_out(randperm(size(y_mat_out,1)),:);
fprintf('\nPrintout of randomly selected sub-set of examples out-of-sample:\n');
fprintf('\nPrediction\tActual');
fprintf('\n----------\t-------');
for i = 1:10
	fprintf('\n%f\t%f\n', y_mat_out(i,1), y_mat_out(i,2));
end

% ===========================================================================
% 6. ANALYSIS OF ERRORS, IN-SAMPLE & OUT-OF-SAMPLE & STATISTICS
% ===========================================================================
confusion_mat_in = confusionMat(predictions_in, y_in);
fprintf('\nConfusion matrix for the predictions and actual class labels in-sample:\n');
disp(confusion_mat_in);

confusion_mat_out = confusionMat(predictions_out, y_out);
fprintf('\nConfusion matrix for the predictions and actual class labels out-of-sample:\n');
disp(confusion_mat_out);

% Let user look at individual class label statistics
while (true)
	viz_stats = yes_or_no('View statistics for individual class predictions? ');
	if (~viz_stats)
		break;
	end
	fprintf('\nChoose which class to print out [%d through %d]:', 1, num_labels);
	chosen_label = input(' ');
	stats_in = classificationStats(confusion_mat_in, chosen_label)
	stats_out = classificationStats(confusion_mat_out, chosen_label)
end

fprintf('\nEnd of program. Bye.\n')
