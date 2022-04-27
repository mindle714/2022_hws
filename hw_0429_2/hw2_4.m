[img, map] = imread('noisy_image.png');

%img_wls = zeros(size(img));
img_wls = [wls(img(:, :, 1), .7, 1.2, 1e-4) ...
    wls(img(:, :, 2), .7, 1.2, 1e-4) ...
    wls(img(:, :, 3), .7, 1.2, 1e-4)];
%img_wls = wlsFilter(im2double(img(1:4,1:4,1)), .7, 1.2);

figure(1)
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
imshow(img_wls);