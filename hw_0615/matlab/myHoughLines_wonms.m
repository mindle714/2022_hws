function [rhos, thetas] = myHoughLines_wonms(H, nLines)
%{
H_flat = reshape(H,1,size(H,1) * size(H,2));
[~, idx] = sort(H_flat, 'descend');
idx = int32(idx);

[r_idx, c_idx] = ind2sub(size(H),idx);
%r_idx = mod((idx-1), size(H,1)) + 1;
%c_idx = idivide((idx-1), size(H,1)) + 1;

thetas = reshape(double(r_idx(1:nLines)), nLines, 1);
rhos = reshape(double(c_idx(1:nLines)), nLines, 1);
%}

peaks = zeros(nLines, 2);
num_peaks = 0;
thres = 0.3*max(H(:));
%thres = 0.5*max(H(:));
window = fix(size(H) / 50);
radius = fix(window / 2);

for i=1:nLines
    [val, idx] = max(H(:));
    if val < thres
        break
    end

    [r_idx, c_idx] = ind2sub(size(H), idx);
    num_peaks = num_peaks + 1;
    peaks(num_peaks, 1) = r_idx;
    peaks(num_peaks, 2) = c_idx;

    H(r_idx, c_idx) = 0;
    %{
    % neighborhood suppression
    for j=r_idx-radius(1):r_idx+radius(1)
        for k=c_idx-radius(2):c_idx+radius(2)
            if (j < 1) || (j > size(H,1)) || (k < 1) || (k > size(H, 2))
                continue
            end
            H(j,k) = 0;
        end
    end
    %}
end

peaks = peaks(1:num_peaks,:);
thetas = peaks(:,1);
rhos = peaks(:,2);