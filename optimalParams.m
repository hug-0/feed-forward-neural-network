function [theta_1, theta_2, J_min, optimal_penalty] = optimalParams(X, y, input_layer_size, hidden_layer_size, num_labels, penalties, MaxIter)
	% OPTIMALPARAMS computes the optimal set of parameters
	% Author: Hugo Nordell
	% Copyright 2016
	random_theta_1 = randomWeights(input_layer_size, hidden_layer_size);
	random_theta_2 = randomWeights(hidden_layer_size, num_labels);
	params_mat = [random_theta_1(:); random_theta_2(:)]*[ones(1, length(penalties))];

	fprintf('\nNumber of penalty terms (lambda) to fit: %d\n', length(penalties));

	% Minimize cost function
	options = optimset('MaxIter', MaxIter);
	J_mat = zeros(MaxIter, length(penalties)); % Store all costs

	% Loop through all penalties to find min(cost)
	for i = 1:length(penalties)
		[params_mat(:,i), J_mat(:,i)] = fmincg(@(t)(cost(t, input_layer_size, hidden_layer_size,...
												 num_labels, X, y, penalties(i))), params_mat(:,i), options);
	end

	% Find smallest cost J as function of penalty at end of iterations
	j_min_col = min(J_mat(end,:), [], 2);

	% Return variable
	optimal_penalty = penalties(j_min_col);

	% Return variable
	J_min = J_mat(:, j_min_col);

	fprintf('\nOptimal lambda (penalty) that minimizes cost J found at: %f', optimal_penalty);
	fprintf('\nComputed cost at this lambda (penalty) is: %f\n', J_min(end));

	% Reshape parameters into matrices before returning

	% Return variable
	theta_1 = reshape(params_mat(1:(input_layer_size+1)*hidden_layer_size,j_min_col),...
										hidden_layer_size, input_layer_size+1);

	% Return variable
	theta_2 = reshape(params_mat(1+(input_layer_size+1)*hidden_layer_size:end,j_min_col),...
										num_labels, hidden_layer_size+1);
end
