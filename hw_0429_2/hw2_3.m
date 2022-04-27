[img, map] = imread('image.png');

img_jb = rolling_guidance(img, 5, 'joint_bilateral', 7, 1., .25);
img_g = rolling_guidance(img, 5, 'guided', 7, 1e-4);

figure(1)
subplot(1,3,1);
imshow(img);
subplot(1,3,2);
imshow(img_jb);
subplot(1,3,3);
imshow(img_g);