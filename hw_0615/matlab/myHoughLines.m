function [rhos, thetas] = myHoughLines(H, nLines)
H_flat = reshape(H,1,size(H,1) * size(H,2));
[~, idx] = sort(H_flat, 'descend');
idx = int32(idx);

r_idx = mod((idx-1), size(H,1)) + 1;
c_idx = idivide((idx-1), size(H,1)) + 1;

thetas = reshape(double(r_idx(1:nLines)), nLines, 1);
rhos = reshape(double(c_idx(1:nLines)), nLines, 1);