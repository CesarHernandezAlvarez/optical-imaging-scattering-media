function thresholded = f_apply_threshold(matrix, threshold)
    thresholded = matrix;
    thresholded(matrix <= threshold) = 0;
end