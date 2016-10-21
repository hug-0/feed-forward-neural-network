function [J, grad] = cost(params,...
													input_layer_size,...
													hidden_layer_size,...
													num_class_labels,...
													X, y, lambda)
	% COST computes the cost for a three layer neural network classifier.
	% Author: Hugo Nordell
	% Copyright 2016

	% ===========================================================================
	% 1. RESHAPE PARAMS INTO MATRICES FOR INPUT AND HIDDEN LAYERS
	% ===========================================================================
	Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
									 hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(params((hidden_layer_size * (input_layer_size + 1) + 1):end), num_class_labels, (hidden_layer_size + 1));

	% ===========================================================================
	% 2. CREATE SHORTHAND UTILITY VARIABLES
	% ===========================================================================
	[m n] = size(X);

	% ===========================================================================
	% 3. SETUP VARIABLES TO RETURN
	% ===========================================================================
	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));

	% ===========================================================================
	% 4. TRANSFORM y TO BE A VECTOR OF m x num_class_labels
	% ===========================================================================
	y_labels = eye(num_class_labels);
	y_mat = y_labels(y,:);

	% ===========================================================================
	% 5. COMPUTE FORWARD PROPAGATION TERMS
	% ===========================================================================
	a1 = [ones(m, 1) X]; % [m x n+1]
	z2 = a1 * Theta1'; % [m x n+1] x [n+1 x hidden]
	a2 = [ones(m, 1) sigmoid(z2)]; % [m x hidden+1]
	z3 = a2 * Theta2'; % [m x hidden+1] x [hidden+1 x num_class_labels]
	a3 = sigmoid(z3); % [m x num_class_labels]

	% ===========================================================================
	% 6. COMPUTE REGULARIZED COST
	% ===========================================================================
	f = y_mat .* log(a3); s = (1 - y_mat) .* log(1 - a3);
	J = -1./m * sum(sum(f + s));
	r = lambda./(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
	J = J + r; % RETURN VARIABLE

	% ===========================================================================
	% 7. COMPUTE REGULARIZED BACKWARD PROPAGATION
	% ===========================================================================
	r1 = lambda./m * [zeros(hidden_layer_size, 1) Theta1(:,2:end)];
	r2 = lambda./m * [zeros(num_class_labels, 1) Theta2(:,2:end)];
	d3 = a3 - y_mat; % [m x num_class_labels]
	d2 = d3 * Theta2 .* a2 .* (1 - a2); % [m x hidden+1]
	d2 = d2(:,2:end); % [m x hidden] Removes bias unit
	Theta1_grad = 1./m * (Theta1_grad + d2' * a1) + r1; % [hidden x n]
	Theta2_grad = 1./m * (Theta2_grad + d3' * a2) + r2; % [num_class_labels x hidden+1]

	% ===========================================================================
	% 8. UNROLL GRADIENTS BEFORE RETURNING
	% ===========================================================================
	grad = [Theta1_grad(:); Theta2_grad(:)]; % RETURN VARIABLE
end
