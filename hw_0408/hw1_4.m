c = imread('input.jpg');
[ce, ~] = HistEq(c);
[ce2, ~] = HistEq_v2(c, 8, 8);
[ce3, ~] = HistEq_v3(c, 8, 8, 4);
%ce3 = adapthisteq(c);

figure(1)
subplot(2,4,1);
imshow(c);

subplot(2,4,2);
imshow(ce);

subplot(2,4,3);
imshow(ce2);

subplot(2,4,4);
imshow(ce3);

subplot(2,4,5);
Hist(c);
ylim([0,7000]);

subplot(2,4,6);
Hist(ce);
ylim([0,7000]);

subplot(2,4,7);
Hist(ce2);
ylim([0,7000]);

subplot(2,4,8);
Hist(ce3);
ylim([0,7000]);