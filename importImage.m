function [img, maps, alpha, info] = importImage(image_path, options)
	% IMPORTIMAGE imports an image and returns a grayscale representation of the image as a [width x height] matrix or as an 'unrolled' matrix (vector)
	% Author: Hugo Nordell
	% Copyright 2016


	% Initial path check
	if (~ischar(image_path))
		error('%s must be a character string.', image_path);
	end

	% Return variable(s)
	[img, maps, alpha] = imread(image_path);
	info = imfinfo(image_path);

	if (exist('options', 'var'))
		if (strcmp(options.GrayScale, 'true'))
			% Convert image to grayscale

			% Using NTSC conversion formula
			img = 0.2989 * img(:,:,1) + 0.5870 * img(:,:,2) + 0.1140 * img(:,:,3);
		end
	else
		% TDB.
	end
end
