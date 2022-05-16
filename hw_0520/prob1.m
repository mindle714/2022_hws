a = [zeros(256, 128) ones(256, 128)];
af = fftshift(fft2(a));

b = zeros(256, 256);
b(78: 178, 78:178) = 1;
bf = fftshift(fft2(b));

[x, y] = meshgrid(1:256, 1:256);
c = (x+y<329) & (x+y>182) & (x-y>-67) & (x-y<73);
cf = fftshift(fft2(c));


figure(1);
subplot(3,2,1);
imshow(a);
subplot(3,2,2);
fftshow(af, 'log');

subplot(3,2,3);
imshow(b);
subplot(3,2,4);
fftshow(bf, 'log');

subplot(3,2,5);
imshow(c);
subplot(3,2,6);
fftshow(cf, 'log');