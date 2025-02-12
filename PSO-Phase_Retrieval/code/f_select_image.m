function [image_0, fig, distance, exposure] = f_select_image(im)
    images = {
        '/MATLAB Drive/Images/d2-25-td.bmp', 'Dots', 2, 25;
        '/MATLAB Drive/Images/d2-30-td.bmp', 'Dots', 2, 30;
        '/MATLAB Drive/Images/d2-35-td.bmp', 'Dots', 2, 35;
        '/MATLAB Drive/Images/d2-40-td.bmp', 'Dots', 2, 40;
        '/MATLAB Drive/Images/d2-45-td.bmp', 'Dots', 2, 45;
        '/MATLAB Drive/Images/d2-50-td.bmp', 'Dots', 2, 50;
        '/MATLAB Drive/Images/d5-25-td.bmp', 'Dots', 5, 25;
        '/MATLAB Drive/Images/d5-30-td.bmp', 'Dots', 5, 30;
        '/MATLAB Drive/Images/d5-35-td.bmp', 'Dots', 5, 35;
        '/MATLAB Drive/Images/d5-40-td.bmp', 'Dots', 5, 40;
        '/MATLAB Drive/Images/d5-45-td.bmp', 'Dots', 5, 45;
        '/MATLAB Drive/Images/d5-50-td.bmp', 'Dots', 5, 50;
        '/MATLAB Drive/Images/d10-25-td.bmp', 'Dots', 10, 25;
        '/MATLAB Drive/Images/d10-30-td.bmp', 'Dots', 10, 30;
        '/MATLAB Drive/Images/d10-35-td.bmp', 'Dots', 10, 35;
        '/MATLAB Drive/Images/d10-40-td.bmp', 'Dots', 10, 40;
        '/MATLAB Drive/Images/d10-45-td.bmp', 'Dots', 10, 45;
        '/MATLAB Drive/Images/d10-50-td.bmp', 'Dots', 10, 50;
        '/MATLAB Drive/Images/l2-25-td.bmp', 'Lines', 2, 25;
        '/MATLAB Drive/Images/l2-30-td.bmp', 'Lines', 2, 30;
        '/MATLAB Drive/Images/l2-35-td.bmp', 'Lines', 2, 35;
        '/MATLAB Drive/Images/l2-40-td.bmp', 'Lines', 2, 40;
        '/MATLAB Drive/Images/l2-45-td.bmp', 'Lines', 2, 45;
        '/MATLAB Drive/Images/l2-50-td.bmp', 'Lines', 2, 50;
        '/MATLAB Drive/Images/l5-25-td.bmp', 'Lines', 5, 25;
        '/MATLAB Drive/Images/l5-30-td.bmp', 'Lines', 5, 30;
        '/MATLAB Drive/Images/l5-35-td.bmp', 'Lines', 5, 35;
        '/MATLAB Drive/Images/l5-40-td.bmp', 'Lines', 5, 40;
        '/MATLAB Drive/Images/l5-45-td.bmp', 'Lines', 5, 45;
        '/MATLAB Drive/Images/l5-50-td.bmp', 'Lines', 5, 50;
        '/MATLAB Drive/Images/l10-25-td.bmp', 'Lines', 10, 25;
        '/MATLAB Drive/Images/l10-30-td.bmp', 'Lines', 10, 30;
        '/MATLAB Drive/Images/l10-35-td.bmp', 'Lines', 10, 35;
        '/MATLAB Drive/Images/l10-40-td.bmp', 'Lines', 10, 40;
        '/MATLAB Drive/Images/l10-45-td.bmp', 'Lines', 10, 45;
        '/MATLAB Drive/Images/l10-50-td.bmp', 'Lines', 10, 50;
        '/MATLAB Drive/Images/N5_gt.tiff', 'Number5', 'N/A', 'N/A';
        '/MATLAB Drive/Images/LR_gt.tiff', 'LetterR', 'N/A', 'N/A';
        '/MATLAB Drive/Images/Resized-GT-Dots.png', 'DotsThesis', 'N/A', 'N/A';
        '/MATLAB Drive/Images/Resized-GT-Lines.png', 'LinesThesis', 'N/A', 'N/A';
    };
    
    if im > 0 && im <= size(images, 1)
        image_0 = imread(images{im, 1});
        fig = images{im, 2};
        distance = images{im, 3};
        exposure = images{im, 4};
    else
        error('Invalid image index');
    end
end