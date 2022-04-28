[low_res, low_res_map] = imread('art_depth_low_res.png');
low_res = ind2rgb(low_res, low_res_map);
[high_res, high_res_map] = imread('art_color_high_res.png');
high_res = ind2rgb(high_res, high_res_map);
[gt, gt_map] = imread('art_depth_gt_high_res.png');
gt = ind2rgb(gt, gt_map);

jb = joint_bilateral(low_res, high_res, 65, 10, 10);
g = guided(low_res, high_res, 65, 4e-3);

figure(1)
subplot(1,4,1);
imshow(low_res);
subplot(1,4,2);
imshow(jb);
subplot(1,4,3);
imshow(g);
subplot(1,4,4);
imshow(gt);

%{
figure(1)
idx = 1;
sigma_s = [1e1];
sigma_r = [1 5 10 15 20];
for i=1:1
    for j=1:5
        subplot(1, 5, idx);
        jb = joint_bilateral(low_res, high_res, 33, sigma_s(i), sigma_r(j));
        imshow(jb);
        diff_jb = mean(abs(gt - jb), 'all');
        title(sprintf('sigma_s:%.2f, sigma_r:%.2f, err:%.4f',sigma_s(i), sigma_r(j), diff_jb));
        idx = idx + 1;
    end
end

figure(1)
idx = 1;
eps = [1e-2 2e-2 4e-2 6e-2 8e-2 1e-3];
for i=1:6
    subplot(1, 6, idx);
    g = guided(low_res, high_res, 65, eps(i));
    imshow(g);
    idx = idx + 1;
end
%}
