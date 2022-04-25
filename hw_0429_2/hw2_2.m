[low_res, low_res_map] = imread('art_depth_low_res.png');
[high_res, high_res_map] = imread('art_color_high_res.png');
[gt, gt_map] = imread('art_depth_gt_high_res.png');

a = joint_bilateral(low_res, high_res, 27, 64., .5);

figure(1)
subplot(1,3,1);
imshow(low_res);
subplot(1,3,2);
imshow(a);
subplot(1,3,3);
imshow(gt);