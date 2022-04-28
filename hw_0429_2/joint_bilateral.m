function output = joint_bilateral(input, guide, ksize, sigma_s, sigma_r)
guide_new_max = 2 * sqrt(ksize^2 + ksize^2);
guide_max = max(guide, [], 'all');
guide = guide / guide_max * guide_new_max;

si = size(input);
output = zeros(si);

radius = fix(ksize/2);
pad_input = padarray(input, [radius, radius], 'symmetric');
pad_guide = padarray(guide, [radius, radius], 'symmetric');
gs = gauss2d(ksize, sigma_s);

for i=1:si(1)
    for j=1:si(2)
        win_input = pad_input(i:i+ksize-1, j:j+ksize-1,:);   
        win_guide = pad_guide(i:i+ksize-1, j:j+ksize-1,:);
        gr = exp(-((win_guide - guide(i, j)).^2) / (2*sigma_r^2));

        for k=1:si(3)
            denom = sum(gs .* gr(:, :, k), 'all');
            output(i, j, k) = sum(gs .* gr(:, :, k) .* win_input(:, :, k), 'all') / denom;
        end
    end
end

function gs = gauss2d(ksize, sigma)
radius = fix(ksize/2);
[X, Y] = meshgrid(-radius:radius, -radius:radius);
gs = (1 / (sigma*sqrt(2*pi))) * exp(-(X.^2+Y.^2) / (2*sigma^2));