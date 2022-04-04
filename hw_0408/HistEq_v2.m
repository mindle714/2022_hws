function [output, sr] = HistEq_v2(input, wtile, htile)
L = cast(intmax(class(input)), 'int16')+1;
sr = 0:1:L-1;

% TODO
wsize = floor(size(input,1) / wtile);
hsize = floor(size(input,2) / htile);
input = input(1:wsize*wtile, 1:hsize*htile);
si = size(input);

% prepare histogram, interpolation and lookup table
hist=zeros(wtile, htile, L);
for i=1:si(1)
    for j=1:si(2)
        hist_i = floor((i-1) / wsize) + 1;
        hist_j = floor((j-1) / hsize) + 1;
        in = input(i,j) + 1;
        hist(hist_i, hist_j, in) = hist(hist_i, hist_j, in) + 1;
    end
end

interp=zeros(si(1), 2);
ind=zeros(si(1), 2);

idxs = double((1:si(1))-1) / wsize - 0.5;
interp(:,1) = idxs - floor(idxs);
interp(:,2) = 1 - interp(:,1);
ind(:,1) = max(floor(idxs), 0) + 1;
ind(:,2) = min(floor(idxs)+1, wtile-1) + 1;

lut=zeros(wtile, htile, L);
for i=1:wtile
    for j=1:htile
        s = 0;
        for k=1:L
            s = s + hist(i, j, k);
            lut(i, j, k) = (s * double(L-1)) / (si(1) * si(2));
        end
    end
end

% interpolate
output = input;
for i=1:si(2)
    idx = double(i-1) / hsize - 0.5;

    interp0 = idx - floor(idx);
    interp1 = 1 - interp0;

    ind0 = max(floor(idx), 0) + 1;
    ind1 = min(floor(idx)+1, htile-1) + 1;

    for j=1:si(2)
        in = input(i,j) + 1;
        output(i,j) = ...
            (lut(ind0, ind(j, 1), in) * interp(j, 2) + ...
            lut(ind0, ind(j, 2), in) * interp(j, 1)) * interp0 + ...
            (lut(ind1, ind(j, 1), in) * interp(j, 2) + ...
            lut(ind1, ind(j, 2), in) * interp(j, 1)) * interp1;
    end
end

size(output);
