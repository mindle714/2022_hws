function Hist(input, print_orig)
%figure(1)
if print_orig
    subplot(1,2,1);
end

histogram(input)
s = size(input);
L = cast(intmax(class(input)), 'int16');
hist=zeros(1, L+1);

for i=1:s(1)
    for j=1:s(2)
        hist(input(i,j)+1) = hist(input(i,j)+1) + 1;
    end
end
subplot(1,2,2);
bar(1:1:length(hist), hist)