function output = rolling_guidance(input, n_iter, g_type, varargin)
input = im2double(input);

if strcmp(g_type, 'joint_bilateral')
    ksize = varargin{1};
    sigma_s = varargin{2};
    sigma_r = varargin{3};
elseif strcmp(g_type, 'guided')
    ksize = varargin{1};
    eps = varargin{2};
else
    error('unsupported guidance type');
end

output = input;
si = size(input);

radius = fix(ksize/2);
pad_input = padarray(input, [radius, radius]);
gs = gauss2d(ksize, 1.);

for i=1:si(1)
    for j=1:si(2)
        win_input = pad_input(i:i+ksize-1, j:j+ksize-1);
        denom = sum(gs, 'all');
        output(i,j) = sum(gs .* win_input, 'all') / denom;
    end
end

prev_output = output;
for iter=1:n_iter
    if strcmp(g_type, 'joint_bilateral')
        output = joint_bilateral(input, prev_output, ksize, sigma_s, sigma_r);
    elseif strcmp(g_type, 'guided')
        output = guided(input, prev_output, ksize, eps);
    end

    prev_output = output;
end

function gs = gauss2d(ksize, a)
radius = fix(ksize/2);
[X, Y] = meshgrid(-radius:radius, -radius:radius);
gs = (1 / (a*sqrt(2*pi))) * exp(-(X.^2+Y.^2)/(2*a^2));