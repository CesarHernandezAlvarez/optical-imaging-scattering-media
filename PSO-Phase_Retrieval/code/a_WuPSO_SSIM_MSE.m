close all;
%% Selection of image for algorithm
for im = 37:38
    clearvars -except im;

    %% Selection of image (1-36)
    [image_0, fig, distance, exposure] = f_select_image(im);
    speckle = double(image_0(:, :, 1));

    %% Downsizing
    [rows, cols] = size(speckle);
    crop_range = rows / 8;
    speckle = speckle(rows/2 - crop_range:rows/2 + crop_range-1, ...
                                cols/2 - crop_range:cols/2 + crop_range-1);
    speckle_gray = mat2gray(f_normalization(speckle));
    autocorrelation = abs(f_FT2Dc(speckle));
    ac_gray = mat2gray(f_normalization(log(autocorrelation)));
    amp_spect_FT_obj = abs(f_FT2Dc(autocorrelation));
    rs_amplitudeFT = sqrt(amp_spect_FT_obj);
    fmbn_gray = mat2gray(f_normalization(log(rs_amplitudeFT)));
    amp_spect_norm = f_normalization(rs_amplitudeFT);
    threshold_bgn = f_min_error(amp_spect_norm);
    amp_spect_at = f_normalization(log(1 + f_apply_threshold(amp_spect_norm, threshold_bgn)));
    fig,imshow(amp_spect_at,[]);
    fm_gray = mat2gray(amp_spect_at);
    
    [rows, cols] = size(amp_spect_at);
    in_ph = zeros(rows,cols);
    row_range = (rows * 15 / 32):(rows * 18 / 32);
    col_range = (cols * 15 / 32):(cols * 18 / 32);
    in_ph(row_range, col_range) = 1;
    
    % pso_iter_list = [10,20,40,50,100,200];
    % part_list = [10,20,30,40,50];
    pso_iter_list = 20;
    part_list = 50;

    for part_i = 1:length(part_list)
        part_no = part_list(part_i);
        for pso_iter_i = 1:length(pso_iter_list)
            pso_iter_no = pso_iter_list(pso_iter_i);
            for rep = 3

                %% Parameters of PSO
                psoParams = f_config_pso(pso_iter_no,part_no);
                psoParams.iter = 0;
                psoParams.update_freq = 200 / pso_iter_no;
                psoParams.VarMin = [1,0.01];
                psoParams.VarMax = [5,0.99];
                psoParams.nVar = 2;

                %% Problem definition
                problem.CostFunction = @(x) wrapperCostFunction(x,in_ph,amp_spect_at,psoParams);
                CostFunction = problem.CostFunction;
                problem.nVar = psoParams.nVar;
                problem.VarMin = psoParams.VarMin;
                problem.VarMax = psoParams.VarMax;

                %% Initialize PSO state
                pso_state = f_initialize_pso(psoParams, CostFunction);
                psoParams.pso_state = pso_state;

                %% Run PSO
                tic;
                fprintf('Image %d. Iterations %d. Particles %d\n', im, psoParams.MaxIt, psoParams.nPop);
                out = f_pso_min(problem, psoParams);
                time_pso = toc;

                BestSol = out.BestSol;
                BestCosts = out.BestCosts;
                BestParticle = out.pop;

                %% Save results and image
                % Retrieve the final gk_at_internal
                [~, gk_at_final] = wrapperCostFunction([], [], [], [], 'get');
                gk_at_final_normalized = f_normalization(gk_at_final);

                %% Saving results
                folder_name_wksp = '/MATLAB Drive/PSO/Special/Workspaces/';
                folder_name_img = '/MATLAB Drive/PSO/Special/Images/';
                if ~isfolder(folder_name_wksp)
                    mkdir(folder_name_wksp);
                end
                if ~isfolder(folder_name_img)
                    mkdir(folder_name_img);
                end
                file_name = char(strcat(fig, '-', int2str(distance), ...
                               'cm-', int2str(exposure), 'ms-', ...
                               int2str(psoParams.MaxIt/psoParams.update_freq), 'it-', ...
                               int2str(psoParams.nPop), 'part-0', ...
                               int2str(rep)));
                file_name_wksp = [folder_name_wksp, file_name,'.mat'];
                file_name_img = [folder_name_img, file_name,'.png'];
                save(file_name_wksp, 'BestSol', 'BestCosts', 'BestParticle', 'psoParams','time_pso');
                imwrite(gk_at_final_normalized, file_name_img);
            end
        end
    end
    general_name = char(strcat(fig, '-', int2str(distance),'cm-', int2str(exposure), 'ms.png'));
    speckle_file = [folder_name_img,'Speckle-',general_name];
    ac_file = [folder_name_img,'AC-',general_name];
    fmbn_file = [folder_name_img,'Fourier-w-noise-',general_name];
    fm_file = [folder_name_img,'Fourier-denoised-',general_name];
    imwrite(speckle_gray,speckle_file)
    imwrite(ac_gray,ac_file)
    imwrite(fmbn_gray,fmbn_file)
    imwrite(fm_gray,fm_file)
end

%% Wrapper Cost Function at the End
function [cost, gk_at_internal_out] = wrapperCostFunction(x, in_ph, amp_spect_at, psoParams, mode)
    persistent gk_at_internal;
    if isempty(gk_at_internal)
        [rows, cols] = size(in_ph);
        gk_at_internal = zeros(rows, cols);
    end

    if nargin == 5 && strcmp(mode, 'get')
        cost = [];
        gk_at_internal_out = gk_at_internal; % Return the internal state
        return;
    end

    % Main cost computation
    [cost, gk_at_new] = f_WuPSO_SSIM_MSE(x, in_ph, amp_spect_at, psoParams, gk_at_internal);
    
    % Update persistent variable for the next iteration
    gk_at_internal = gk_at_new;
    gk_at_internal_out = [];
end
