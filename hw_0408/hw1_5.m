c = imread('input.jpg');
cm = imread('input_match.png');
ce = HistMatching(c, cm);

figure(1)
subplot(2,3,1);
imshow(c);

subplot(2,3,2);
imshow(cm);

subplot(2,3,3);
imshow(ce);

subplot(2,3,4);
Hist(c);

subplot(2,3,5);
Hist(cm);

subplot(2,3,6);
Hist(ce);







 

