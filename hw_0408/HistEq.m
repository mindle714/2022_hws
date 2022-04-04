function [output, sr] = HistEq(input)
output = input;
si = size(input);
L = cast(intmax(class(input)), 'int16')+1;
sr = 0:1:L-1;
hist=zeros(1, L);

for i=1:si(1)
    for j=1:si(2)
        hist(input(i,j)+1) = hist(input(i,j)+1) + 1;
    end
end

for i=1:length(sr)
    s = 0;
    for j=1:i
        s = s + hist(j);
    end
    sr(i) = (s * double(L-1)) / (si(1) * si(2));
    sr(i) = cast(sr(i), class(input));
end

for i=1:si(1)
    for j=1:si(2)
        output(i,j) = sr(input(i,j)+1);
    end
end
