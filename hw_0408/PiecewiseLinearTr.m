function output = PiecewiseLinearTr(input,a,b) %
% PiecewiseLinearTr(IM,A,B) applies a piecewise linear transformation to the pixel values
% of the input image INPUT, where A and B are vectors containing the x and y coordinates
% of the ends of the line segments. INPUT can be of type DOUBLE,
% and the values in A and B must be between 0 and 1 (normalized intensity values). %
% For example:
%
% PiecewiseLinearTr(x,[0,1],[1,0])
%
% simply do negative transform inverting the pixel values.
%

if length(a) ~= length (b)
    error('Vectors A and B must be of equal size');
end

if length(a) < 2
    error('Length of A and B must be bigger or equal to 2');
end

s = size(input);
output = zeros(s);

linear = zeros(length(a), 2);
for i = 1:(length(linear)-1)
    linear(i,1) = (b(i+1) - b(i)) / (a(i+1) - a(i));
    linear(i,2) = (a(i+1) * b(i) - a(i) * b(i+1)) / (a(i+1) - a(i));
end

for i = 1:s(1)
    for j = 1:s(2)
        for k = 1:(length(a)-1)
            if (input(i,j) >= a(k)) && (input(i,j) <= a(k+1))
                output(i,j) = linear(k,1) * input(i,j) + linear(k,2);
            end
        end
    end
end