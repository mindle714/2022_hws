[img, map] = imread('noisy_image.png');
img = im2double(img);

img_wls = zeros(size(img));
img_wls(:, :, 1) = wls(img(:, :, 1), 2, 1.2, 1e-4);
img_wls(:, :, 2) = wls(img(:, :, 2), 2, 1.2, 1e-4);
img_wls(:, :, 3) = wls(img(:, :, 3), 2, 1.2, 1e-4);
img_bl = bilateral(img, 11, 5);
img_g = guided(img, img, 17, 1e-1);

figure(1)
subplot(1,4,1);
imshow(img);
subplot(1,4,2);
imshow(img_bl);
subplot(1,4,3);
imshow(img_g);
subplot(1,4,4);
imshow(img_wls);