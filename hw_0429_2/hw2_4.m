[img, map] = imread('noisy_image.png');

img_wls = wls(img(:,:,1), .35, 1.2, 1e-4);
%img_wls = wlsFilter(im2double(img(:,:,1)), .7, 1.2);

figure(1)
subplot(1,2,1);
imshow(img(:,:,1));
subplot(1,2,2);
imshow(img_wls);