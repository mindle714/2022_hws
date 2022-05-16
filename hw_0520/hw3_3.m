img1 = imread("mandrill.tif");
img1 = im2double(rgb2gray(img1));

img2 = imread("clown.tif");
img2 = im2double(img2);

f_img1 = fftshift(fft2(img1));
mag_img1 = abs(f_img1);
phase_img1 = angle(f_img1); % atan2(imag(f_img1), real(f_img1));

f_img2 = fftshift(fft2(img2));
mag_img2 = abs(f_img2);
phase_img2 = angle(f_img2); % atan2(imag(f_img2), real(f_img2));

f_img1_p2 = mag_img1 .* (cos(phase_img2) + sin(phase_img2) .* 1j);
img1_p2 = real(ifft2(ifftshift(f_img1_p2)));

f_img2_p1 = mag_img2 .* (cos(phase_img1) + sin(phase_img1) .* 1j);
img2_p1 = real(ifft2(ifftshift(f_img2_p1)));

figure(1)
subplot(2,2,1);
imshow(img1);

subplot(2,2,2);
imshow(img2);

subplot(2,2,3);
imshow(img1_p2);

subplot(2,2,4);
imshow(img2_p1);

f_img1_trans = fftshift(fft2(img1_p2));
mag_img1_trans = abs(f_img1_trans);
phase_img1_trans = angle(f_img1_trans);

f_img2_trans = fftshift(fft2(img2_p1));
mag_img2_trans = abs(f_img2_trans);
phase_img2_trans = angle(f_img2_trans);

figure(2)
subplot(4,3,1);
imshow(img1);

subplot(4,3,2);
%imagesc(log(mag_img1));
%colormap(gray);
fftshow(mag_img1, 'log');

subplot(4,3,3);
imagesc(phase_img1);
colormap(gray);

subplot(4,3,4);
imshow(img2);

subplot(4,3,5);
%imagesc(log(mag_img2));
%colormap(gray);
fftshow(mag_img2, 'log');

subplot(4,3,6);
imagesc(phase_img2);
colormap(gray);

subplot(4,3,7);
imshow(img1_p2);

subplot(4,3,8);
%imagesc(log(mag_img1_trans));
%colormap(gray);
fftshow(mag_img1_trans, 'log');

subplot(4,3,9);
imagesc(phase_img1_trans);
colormap(gray);

subplot(4,3,10);
imshow(img2_p1);

subplot(4,3,11);
%imagesc(log(mag_img2_trans));
%colormap(gray);
fftshow(mag_img2_trans, 'log');

subplot(4,3,12);
imagesc(phase_img2_trans);
colormap(gray);