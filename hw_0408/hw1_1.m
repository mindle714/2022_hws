c = im2double(imread('cameraman.tif'));

cTr = PiecewiseLinearTr(c, [0,1], [1,0]);
cTr2 = PiecewiseLinearTr(c, [0 .25 .5 .75 1],[0 .75 .25 .5 1]);

figure(1)
subplot(1,3,1);
imshow(c);
subplot(1,3,2);
imshow(cTr);
subplot(1,3,3);
imshow(cTr2);






 

