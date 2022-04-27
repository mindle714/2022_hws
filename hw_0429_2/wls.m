function output = wls(input, lambda, alpha, eps)
input = im2double(input);

si = size(input);
si_all = si(1) * si(2);

dl_dx = diff(log(max(input, eps)), 1, 1);
dl_dx = padarray(dl_dx, [1, 0], 'post');
a_x = abs(dl_dx).^alpha + eps;
a_x = 1. ./ a_x;
a_x = a_x(:);

a_x_shift = padarray(a_x, si(1), 'pre');
a_x_shift = a_x_shift(1:size(a_x));

dl_dy = diff(log(max(input, eps)), 1, 2);
dl_dy = padarray(dl_dy, [0, 1], 'post');
a_y = abs(dl_dy).^alpha + eps;
a_y = 1. ./ a_y;
a_y = a_y(:);

a_y_shift = padarray(a_y, 1, 'pre');
a_y_shift = a_y_shift(1:size(a_y));

lg_diag = a_x + a_x_shift + a_y + a_y_shift;
lg_diag = lg_diag + 1 / lambda;
lg = spdiags([a_x, a_y, lg_diag, a_y, a_x], [-si(1), -1, 0, 1, si(1)], si_all, si_all);
lhs = lambda * lg;%eye(size(lg)) + lambda * lg;
rhs = input(:);

output = lhs \ rhs;
output = reshape(output, size(input));
