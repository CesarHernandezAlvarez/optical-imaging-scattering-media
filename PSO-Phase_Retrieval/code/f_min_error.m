function thresh_val = f_min_error(input_image)
    % Normalize and preprocess input image
    if max(input_image(:)) == 1
        input_image = im2uint8(input_image); % Scale image to 8-bit
    end
    
    % Compute histogram
    hist_vals = imhist(input_image);
    N = 256; 
    total_pixels = sum(hist_vals);
    
    % Initial threshold guesses
    initial_guesses = [N / 8, N / 4, N / 2, 3 * N / 4]; % Multiple initial guesses
    epsilon = 1e-6; % Convergence tolerance
    thresh_val = -1; % Invalid threshold initialization
    
    % Iterate through different initial guesses
    for guess = initial_guesses
        prev_t = 0;
        curr_t = guess;

        % Iterate to find optimal threshold
        while abs(curr_t - prev_t) > epsilon
            prev_t = curr_t;
            
            % Calculate probabilities, means, and variances for both regions
            % Region 1 (below threshold)
            hist_1 = hist_vals(1:curr_t);
            p1 = sum(hist_1) / total_pixels;
            if p1 == 0, break; end
            u1 = sum((1:curr_t)' .* hist_1) / (total_pixels * p1);
            o1 = sum(((1:curr_t)'.^2) .* hist_1) / (total_pixels * p1) - u1^2;

            % Region 2 (above threshold)
            hist_2 = hist_vals(curr_t+1:end);
            p2 = sum(hist_2) / total_pixels;
            if p2 == 0, break; end
            u2 = sum((curr_t+1:N)' .* hist_2) / (total_pixels * p2);
            o2 = sum(((curr_t+1:N)'.^2) .* hist_2) / (total_pixels * p2) - u2^2;

            % Check for invalid variances
            if o1 <= 0 || o2 <= 0
                break;
            end

            % Calculate new threshold using quadratic formula
            A = 1 / o1 - 1 / o2;
            B = u1 / o1 - u2 / o2;
            C = u1^2 / o1 - u2^2 / o2 + log((p2^2 * o1) / (p1^2 * o2));

            % Check for valid quadratic solution
            if B^2 < A * C || A == 0
                break;
            end

            curr_t = floor((-B + sqrt(B^2 - A * C)) / A);

            % Check for valid threshold
            if curr_t > 0 && curr_t < N
                thresh_val = curr_t / (N - 1);
                return;  % Exit function with valid threshold
            end
        end
    end

    % If no valid threshold found, set a default value (e.g., mean threshold)
    if thresh_val < 0
        warning('No valid threshold found, setting to default mean value.');
        thresh_val = (mean(input_image(:)) / 255)*1.5;
    end
end
