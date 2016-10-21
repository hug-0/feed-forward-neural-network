function [C, order] = confusionMat(predictions, actuals)
	% CONFUSIONMAT computes and returns the confusion matrix for multi-class classification precision and recall.
	% Author: Hugo Nordell
	% Copyright 2016

	% Unique elements in actuals
	labels = unique(actuals);
	order = labels;

	% Confusion matrix
	C = zeros(length(labels), length(labels));

	% Fill matrix
	for i = 1:length(labels)
		for j = 1:length(labels)
			if (j)
				C(i,j) = sum(predictions(find(actuals == i)) == j);
			end
		end
	end
end
