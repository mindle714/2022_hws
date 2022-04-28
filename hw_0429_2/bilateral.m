function output = bilateral(input, ksize, sigma_s)
si = size(input);
output = zeros(si);

radius = fix(ksize/2);
pad_input = padarray(input, [radius, radius], 'symmetric');
gs = gauss2d(ksize, sigma_s);

for i=1:si(1)
    for j=1:si(2)
        win_input = pad_input(i:i+ksize-1, j:j+ksize-1,:);   
        for k=1:si(3)
            denom = sum(gs, 'all');
            output(i, j, k) = sum(gs .* win_input(:, :, k), 'all') / denom;
        end
    end
end

function gs = gauss2d(ksize, sigma)
radius = fix(ksize/2);
[X, Y] = meshgrid(-radius:radius, -radius:radius);
gs = (1 / (sigma*sqrt(2*pi))) * exp(-(X.^2+Y.^2) / (2*sigma^2));