function Hist(input)

histogram(input)
s = size(input);
L = cast(intmax(class(input)), 'int16');
hist=zeros(1, L+1);

for i=1:s(1)
    for j=1:s(2)
        hist(input(i,j)+1) = hist(input(i,j)+1) + 1;
    end
end

bar(1:1:length(hist), hist)