[low_res, low_res_map] = imread('art_depth_low_res.png');
low_res = ind2rgb(low_res, low_res_map);
[high_res, high_res_map] = imread('art_color_high_res.png');
high_res = ind2rgb(high_res, high_res_map);
[gt, gt_map] = imread('art_depth_gt_high_res.png');
gt = ind2rgb(gt, gt_map);

jb = joint_bilateral(low_res, high_res, 65, 10, 10);
g = guided(low_res, high_res, 33, 1e-5);

figure(1)
subplot(2,2,1);
imshow(low_res);
subplot(2,2,2);
imshow(jb);
subplot(2,2,3);
imshow(g);
subplot(2,2,4);
imshow(gt);