[img, map] = imread('image.png');
img = im2double(img);

img_jbs = rolling_guidance(img, 5, 'joint_bilateral', 25, 10, 10);
img_gs = rolling_guidance(img, 5, 'guided', 17, 4e-3);

figure(1)

idx = 1;
for img_idx=[1 2 4 6]
    subplot(2,4,idx);
    imshow(squeeze(img_jbs(img_idx,:,:,:)));
    idx = idx + 1;
end

for img_idx=[1 2 4 6]
    subplot(2,4,idx);
    imshow(squeeze(img_gs(img_idx,:,:,:)));
    idx = idx + 1;
end