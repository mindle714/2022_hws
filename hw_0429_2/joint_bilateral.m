function output = joint_bilateral(input, guide, ksize, sigma_s, sigma_r)
input = im2double(input);
guide = im2double(guide);

output = input;
si = size(input);

radius = fix(ksize/2);
pad_input = padarray(input, [radius, radius]);
pad_guide = padarray(guide, [radius, radius]);
gs = gauss2d(ksize, sigma_s);

for i=1:si(1)
    for j=1:si(2)
        win_input = pad_input(i:i+ksize-1, j:j+ksize-1);
        win_guide = pad_guide(i:i+ksize-1, j:j+ksize-1);
        win_guide = abs(win_guide - guide(i, j));
        win_guide = exp(-1/2 * win_guide.^2 / (sigma_r^2));

        denom = sum(gs .* win_guide, 'all');
        output(i,j) = sum(gs .* win_guide .* win_input, 'all') / denom;
    end
end

function gs = gauss2d(ksize, a)
radius = fix(ksize/2);
[X, Y] = meshgrid(-radius:radius, -radius:radius);
gs = (1 / (a*sqrt(2*pi))) * exp(-(X.^2+Y.^2)/(2*a^2));