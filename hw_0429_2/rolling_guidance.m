function outputs = rolling_guidance(input, n_iter, g_type, varargin)
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

si = size(input);
output = zeros(si);

radius = fix(ksize/2);
pad_input = padarray(input, [radius, radius]);
gs = gauss2d(ksize, 4);

for i=1:si(1)
    for j=1:si(2)
        for k=1:si(3)
            win_input = pad_input(i:i+ksize-1, j:j+ksize-1, :);
            denom = sum(gs, 'all');
            output(i, j, k) = sum(gs .* win_input(:, :, k), 'all') / denom;
        end
    end
end

outputs = zeros(n_iter+1, si(1), si(2), si(3));
prev_output = output;
outputs(1,:,:,:) = output;

for iter=1:n_iter
    if strcmp(g_type, 'joint_bilateral')
        output = joint_bilateral(input, prev_output, ksize, sigma_s, sigma_r);
    elseif strcmp(g_type, 'guided')
        output = guided(input, prev_output, ksize, eps);
    end

    prev_output = output;
    outputs(iter+1,:,:,:) = output;
end

function gs = gauss2d(ksize, sigma)
radius = fix(ksize/2);
[X, Y] = meshgrid(-radius:radius, -radius:radius);
gs = (1 / (sigma*sqrt(2*pi))) * exp(-(X.^2+Y.^2) / (2*sigma^2));