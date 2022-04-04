c = imread('input.jpg');
[ce, ~] = HistEq(c);

figure(1)
subplot(2,2,1);
imshow(c);

subplot(2,2,2);
imshow(ce);

subplot(2,2,3);
Hist(c);

subplot(2,2,4);
Hist(ce);