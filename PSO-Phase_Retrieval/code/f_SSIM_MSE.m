%% SSIM for combining with MSE
function loss_final = f_SSIM_MSE(imfileR,imfileG)
    % imfileR = imread('Lines-10cm-40ms-30-Correction.jpg');
    % imfileG = imread('Lines-10cm-45ms-29-Correction.jpg');
%     figure,imshow(imfileR,[]);
%     figure,imshow(imfileG,[]);
    imR = im2double(imfileR);
    imG = im2double(imfileG);
    n = numel (imR);
    MSE_RG = (1/n)*sum((imR-imG).^2,'all');
    meanR = mean(imR,'all');
    meanG = mean(imG,'all');
    varR = var(imR,0,'all');
    varG = var(imG,0,'all');
    covRG = mean(cov(imR,imG),'all');
    L = 255;
    k1 = 0.01;
    k2 = 0.03;
    c1 = (k1*L)^2;
    c2 = (k2*L)^2;
    SSIM_RG = ((2*(meanR)*(meanG)+c1)*(2*(covRG)+c2))/(((meanR^2)*(meanG^2)+c1)*((varR)*(varG)+c2));
    balance_loss = 10;
    loss_final = (1-SSIM_RG)+balance_loss*MSE_RG;
end