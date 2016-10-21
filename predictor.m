function [p] = predictor(params1, params2, X)
	% PREDICTOR computes the predicted class output for a 3 layer neural network with parameters param1 (input), param2 (hidden) and features X
	% Author: Hugo Nordell
	% Copyright 2016

	h1 = sigmoid([ones(size(X),1) X] * params1');
	h2 = sigmoid([ones(size(X),1) h1] * params2');
	[garbage p] = max(h2, [], 2);
end
