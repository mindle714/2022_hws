c = imread('input.jpg');
%class(c)
[ce, sr] = HistEq(c);
[ce2, sr] = HistEq_v2(c, 8, 8);
[ce3, sr] = HistEq_v3(c, 8, 8, 40);
figure(1)
subplot(1,4,1);
imshow(c);
subplot(1,4,2);
imshow(ce);
subplot(1,4,3);
imshow(ce2);
subplot(1,4,4);
imshow(ce3);

figure(2);
Hist(c,false);
figure(3);
Hist(ce,false);







 

