c = imread('input.jpg');
%class(c)
[ce, sr] = HistEq(c);
figure(1)
subplot(1,2,1);
imshow(c);
subplot(1,2,2);
imshow(ce);

figure(2);
Hist(c,false);
figure(3);
Hist(ce,false);







 

