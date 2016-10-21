function [grad] = sigmoidGradient(z)
	% SIGMOIDGRADIENT takes a sigmoid z (scalar, vector or matrix) and returns the gradient (scalar, vector or matrix)
	grad = sigmoid(z).*(1-sigmoid(z));
end
