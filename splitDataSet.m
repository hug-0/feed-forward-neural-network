function [in_sample, cross_val, out_sample] = splitDataSet(data, training_size_ratio, cv_size_ratio)
	% SPLITDATASET takes a matrix or vector as input together with preferred ratios for splitting a data set into a training set and cross-validation set and returns up to three vectors or matrices of the split data.
	% Author: Hugo Nordell
	% Copyright 2016

	if (training_size_ratio + cv_size_ratio >= 1)
		error('Ratios to split data into cannot be >= 1.');
	end

	in_sample = data(1:length(data)*training_size_ratio,:);
	cross_val = data(length(in_sample)+1:end-length(data)*(1-training_size_ratio-cv_size_ratio),:);
	out_sample = data((length(in_sample)+length(cross_val)+1):end,:);
end
