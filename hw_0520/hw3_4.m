img = imread("pattern.tif");
img = im2double(img);

f_img = fftshift(fft2(img));
mag_img = abs(f_img);
phase_img = angle(f_img);

figure(1)
subplot(2,2,1);
imshow(img);

subplot(2,2,2);
%imagesc(log(mag_img));
%colormap(gray);
fftshow(mag_img, 'log');

filt_spatial = zeros(size(mag_img));
filt_spatial(129, :) = 1;
filt_spatial(:, 129) = 1;
filt_spatial(120:138, 120:138) = 0;

%{
filt_idxs = log(mag_img) > 6;
filt_idxs = logical(filt_idxs .* filt_spatial);
mag_img_filt = mag_img;
mag_img_filt(filt_idxs) = 0;
%}

mag_img_filt = (1-filt_spatial) .* mag_img;

f_img_filt = mag_img_filt .* (cos(phase_img) + sin(phase_img) .* 1j);
img_filt = real(ifft2(ifftshift(f_img_filt)));

subplot(2,2,3);
imshow(img_filt);

subplot(2,2,4);
%imagesc(log(mag_img_filt));
%colormap(gray);
fftshow(mag_img_filt, 'log');

img_filt_sub = img - img_filt;
f_img_filt_sub = fftshift(fft2(img - img_filt));
mag_img_filt_sub = abs(f_img_filt_sub);

figure(2)
subplot(3,2,1);
imshow(img);

subplot(3,2,2);
%imagesc(log(mag_img));
%colormap(gray);
fftshow(mag_img, 'log');

img3 = imread("pattern_v3.tif");
img3 = im2double(rgb2gray(img3(:,:,1:3)));
f_img3 = fftshift(fft2(img3));
mag_img3 = abs(f_img3);
phase_img3 = angle(f_img3);

subplot(3,2,3);
imshow(img3);

subplot(3,2,4);
%imagesc(log(mag_img3));
%colormap(gray);
fftshow(mag_img3, 'log');

img2 = 1.-(img3-img);
f_img2 = fftshift(fft2(img2));
mag_img2 = abs(f_img2);
phase_img2 = angle(f_img2);

subplot(3,2,5);
imshow(img2);

subplot(3,2,6);
%imagesc(log(mag_img2));
%colormap(gray);
fftshow(mag_img2, 'log');

figure(3)
subplot(2,3,1);
%imagesc(log(mag_img));
%colormap(gray);
fftshow(mag_img, 'log');

subplot(2,3,2);
%imagesc(log(mag_img - mag_img_filt));
%colormap(gray);
fftshow(mag_img - mag_img_filt, 'log');

subplot(2,3,3);
%imagesc(log(mag_img_filt));
%colormap(gray);
fftshow(mag_img_filt, 'log');

subplot(2,3,4);
%imagesc(log(mag_img));
%colormap(gray);
fftshow(mag_img, 'log');

subplot(2,3,5);
%imagesc(log(mag_img3));
%colormap(gray);
fftshow(mag_img3, 'log');

subplot(2,3,6);
%imagesc(log(mag_img2));
%colormap(gray);
fftshow(mag_img2, 'log');