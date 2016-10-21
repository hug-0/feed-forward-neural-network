function plotData(X, y)
	% PLOTDATA plots the features against the class labels and the features against each other in 2D.
	% Author: Hugo Nordell
	% Copyright 2016

	% Split y into all unique classes
	labels = unique(y)';

	% Plot features against class labels
	for i = 1:length(X(1,:))
		figure(i);
		plot(X(:,i), y(:), 'ro');
		xlabel(strcat('X_',num2str(i),' ',' in-sample'));
		ylabel('y in-sample');
	end

	% Plot features against each other
	markers = 'ox*.+<>^sdv';
	colors = 'rgbcpm';
	if (length(markers) < length(labels))
		warning('There aren\t enough marker types available to uniquely plot all class labels. Please consider another form of visualization.');
	end
	if (length(colors) < length(labels))
		warning('There aren\t enough colors to distinctly plot all class labels. Please consider another form of visualization.');
	end

	for i=1:length(X(1,:))
		for j=i:length(X(1,:))
			if (i ~= j)
				figure();
				for k = labels
					plot(X(find(y == k),j), X(find(y == k),i), 'LineStyle', 'None', 'Marker', markers(k), 'Color', colors(k), 'MarkerSize', 10);
					hold on;
				end
				xlabel(strcat('X_',num2str(i),' ',' in-sample'));
				ylabel(strcat('X_',num2str(j),' ',' in-sample'));
			end
		end
	end
end
