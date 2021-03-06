function [output, lut] = HistEq(input)
output = input;
si = size(input);
L = cast(intmax(class(input)), 'int16')+1;
lut = 0:1:L-1;
hist=zeros(1, L);

for i=1:si(1)
    for j=1:si(2)
        hist(input(i,j)+1) = hist(input(i,j)+1) + 1;
    end
end

s = 0;
for i=1:L
    s = s + hist(i);
    lut(i) = (s * double(L-1)) / (si(1) * si(2));
    lut(i) = cast(lut(i), class(input));
end

for i=1:si(1)
    for j=1:si(2)
        output(i,j) = lut(input(i,j)+1);
    end
end
