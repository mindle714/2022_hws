img = imread("mandrill.tif");
img = im2double(rgb2gray(img));

f_img = fftshift(fft2(img));
mag_img = abs(f_img);
phase_img = angle(f_img); % atan2(imag(f_img), real(f_img));

figure(1)
subplot(1,3,1);
imshow(img);

subplot(1,3,2);
imagesc(log(mag_img));
colormap(gray);

subplot(1,3,3);
imagesc(phase_img);
colormap(gray);