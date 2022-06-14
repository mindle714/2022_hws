function [thres_mag] = myEdgeFilter_wonms(img0, sigma)
si = size(img0);
img1 = zeros(si);

hsize = 2*ceil(3*sigma) + 1;
gs = gauss2d(hsize, sigma);
gs = gs / sum(gs, 'all');

radius = fix(hsize/2);
pad_img0 = padarray(img0, [radius, radius], 'symmetric');

for i=1:si(1)
    for j=1:si(2)
        win_img0 = pad_img0(i:i+hsize-1, j:j+hsize-1);   
        img1(i, j) = sum(gs .* win_img0, 'all');
    end
end

sobel_x = [-1 -2 -1; 0 0 0; 1 2 1];
sobel_y = [-1 0 1; -2 0 2; -1 0 1];

img2_x = zeros(si);
img2_y = zeros(si);
pad_img1 = padarray(img1, [1, 1], 'symmetric');

for i=1:si(1)
    for j=1:si(2)
        win_img1 = pad_img1(i:i+2, j:j+2);
        img2_x(i, j) = sum(sobel_x .* win_img1, 'all');
        img2_y(i, j) = sum(sobel_y .* win_img1, 'all');
    end
end

sobel_mag = sqrt(img2_x.^2 + img2_y.^2);
sobel_ph = atan2(img2_y, img2_x);
pad_sobel_mag = padarray(sobel_mag, [1, 1], 0);
nms_mag = sobel_mag;

%{
for i=1:si(1)
    for j=1:si(2)
        win_sobel_mag = pad_sobel_mag(i:i+2, j:j+2);
        ph = sobel_ph(i,j);
        mag = sobel_mag(i,j);

        % rotate by 90 deg is considered
        if ((ph >= deg2rad(-22.5)) && (ph < deg2rad(22.5))) || ((ph >= deg2rad(157.5)) || (ph < deg2rad(-157.5)))
            if (win_sobel_mag(3,2) > mag) || (win_sobel_mag(1,2) > mag)
                nms_mag(i,j) = 0;
            end

        elseif ((ph >= deg2rad(22.5)) && (ph < deg2rad(67.5))) || ((ph >= deg2rad(-157.5)) && (ph < deg2rad(-112.5)))
            if (win_sobel_mag(3,3) > mag) || (win_sobel_mag(1,1) > mag)
                nms_mag(i,j) = 0;
            end

        elseif ((ph >= deg2rad(67.5)) && (ph < deg2rad(112.5))) || ((ph >= deg2rad(-112.5)) && (ph < deg2rad(-67.5)))
            if (win_sobel_mag(2,1) > mag) || (win_sobel_mag(2,3) > mag)
                nms_mag(i,j) = 0;
            end

        elseif ((ph >= deg2rad(112.5)) && (ph < deg2rad(157.5))) || ((ph >= deg2rad(-67.5)) && (ph < deg2rad(-22.5)))
            if (win_sobel_mag(3,1) > mag) || (win_sobel_mag(1,3) > mag)
                nms_mag(i,j) = 0;
            end
        end
    end
end
%}

thres_mag = nms_mag;
%{
min_thres = 0.08;
max_thres = 0.2;
thres_mag = (nms_mag > max_thres);
pad_thres_mag = padarray(thres_mag, [1, 1], 0);

for i=1:si(1)
    for j=1:si(2)
        win_thres_mag = pad_thres_mag(i:i+2, j:j+2);
        near_edge = (sum(win_thres_mag, 'all') > 0);
        if (thres_mag(i,j) >= min_thres) && (thres_mag(i,j) <= max_thres)
            if near_edge
                thres_mag(i,j) = 1;
            end
        end
    end
end
%}
    
function gs = gauss2d(ksize, sigma)
radius = fix(ksize/2);
[X, Y] = meshgrid(-radius:radius, -radius:radius);
gs = (1 / (sigma*sqrt(2*pi))) * exp(-(X.^2+Y.^2) / (2*sigma^2));