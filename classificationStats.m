function [stats] = classificationStats(cm, class_label)
	% CLASSIFICATIONSTATS returns a struct object with statistics about a given classification problem and specific class label.
	% Input: a confusion matrix of size N x N, i.e. a contingency table
	% Output: struct with statistics for a given class label
	% Author: Hugo Nordell
	% Copyright 2016

	% Return variable
	stats = {};

	% Shorthand notations of key metrics
	TP = cm(class_label, class_label);
	FP = sum(cm(:,class_label)) - TP;
	FN = sum(cm(class_label,:), 2) - TP;
	TN = sum(sum(cm)) - TP - FP - FN;

	stats.true_positive = TP;
	stats.false_positive = FP;
	stats.true_negative = TN;
	stats.false_negative = FN;

	% F1 Score
	stats.F1_score = (2*TP)/(2*TP + FP + FN);
	% Matthews Correlation Coefficient
	stats.matthews_corr_coeff = (TP*TN - FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
	% Accuracy
	stats.accuracy = (TP + TN)/(sum(sum(cm)));
	% Precision
	stats.precision = (TP)/(TP + FP);
	% Recall
	stats.recall = (TP)/(TP + FN);
	% Specificity
	stats.specificity = (TN)/(FP + TN);
	% Negative Predictive Value
	stats.negative_predictive_value = (TN)/(TN + FN);
	% Fall-out / False Positive Rate
	stats.false_positive_rate = (FP)/(FP + TN);
	% False Discovery Rate
	stats.false_discovery_rate = (FP)/(FP + TP);
	% Miss Rate / False Negative Rate
	stats.false_negative_rate = (FN)/(FN + TP);
end
