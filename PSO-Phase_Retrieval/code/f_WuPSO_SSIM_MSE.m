function [outError, gk_at] = f_WuPSO_SSIM_MSE(x, in_ph, amp_spect_at, psoParams, gk_at_new)

    % Update PSO state every update_freq iterations
    if mod(psoParams.iter, psoParams.update_freq) == 0 && psoParams.iter ~= 0
        psoParams.pso_state = f_pso_update(psoParams.pso_state,iter);
    end

    % Extract parameters from x
    sigmaGF = x(1);
    threshold = x(2);

    if psoParams.iter == 0
        object_estimation = amp_spect_at .* exp(1i * in_ph);
    else
        object_estimation = gk_at_new;
    end

    % Perform object estimation
    Gku = f_FT2Dc(object_estimation);
    Gku_prime = amp_spect_at .* exp(1i * angle(Gku));
    gkx = real(f_IFT2Dc(Gku_prime));
    gk_bt = max(gkx, 0);
    gk_agf = imgaussfilt(abs(gk_bt), sigmaGF);

    % Thresholding
    gk_agf = f_normalization(gk_agf);
    gk_at = f_apply_threshold(gk_agf, threshold);

    % SSIM/MSE Error Calculation
    outError = f_SSIM_MSE(f_normalization(real(f_FT2Dc(gk_at))), f_normalization(real(amp_spect_at)));
    
    % Debug: Print parameters and error
    % fprintf('sigmaGF: %f, threshold: %f, outError: %f\n', sigmaGF, threshold, outError);

    % fprintf('Gku sum: %.6f, Gku_prime sum: %.6f, gk_bt sum: %.6f, gk_agf sum: %.6f\n', ...
    % sum(Gku(:)), sum(Gku_prime(:)), sum(gk_bt(:)), sum(gk_agf(:)));
end
