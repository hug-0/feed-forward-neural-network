function [weights] = randomWeights(L_in, L_out, options)
	% RANDOMWEIGHTS creates a random set of initial weights for the parameters of a neural network layer with L_in ingoing units and L_out outgoing units.
	% The options argument determines if a bias unit should be added, e.g. options.bias = 'true'. Default is true.
	% Author: Hugo Nordell
	% Copyright 2016

	% Return variable
	weights = zeros(L_out, L_in + 1);

	if (exist('options', 'var'))
		if (~isfield(options, 'Bias'))
			warning('No Bias option set. Will assume bias to be added.');
		elseif (strcmp(options.Bias, 'false'))
			weights = zeros(L_out, L_in);
		end
	end
	eps_init =  sqrt(6)./sqrt(L_in + L_out);
	% Randomize weights between [-eps, eps]
	weights = rand(L_out, L_in + 1) * 2 * eps_init - eps_init;
end
