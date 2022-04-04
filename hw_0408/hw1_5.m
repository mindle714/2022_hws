c = imread('input.jpg');
cm = imread('input_match.png');
%class(c)
ce = HistMatching(c, cm);
figure(1)
subplot(1,2,1);
imshow(c);
subplot(1,2,2);
imshow(ce);

figure(2);
Hist(c,false);
figure(3);
Hist(ce,false);







 

