function output = HistMatching(input, target)
output = input;
si = size(input);

[~, sr1] = HistEq(input);
[~, sr2] = HistEq(target);
sr2inv = zeros(1, length(sr2));

for i=1:length(sr2)
    sr2inv(sr2(i)+1) = i-1;
end

lhs = sr2inv(1); width = 1;
if sr2inv(length(sr2inv)) == 0
    sr2inv(length(sr2inv)) = length(sr2inv)-1;
end

for i=2:length(sr2inv)
    if sr2inv(i) == 0
        width = width + 1;
    else
        rhs = sr2inv(i);
        if width > 1
            for j=1:width
                sr2inv(i-width+j) = lhs + ((rhs-lhs) * j/width);
                sr2inv(i-width+j) = cast(sr2inv(i-width+j), 'int16');
            end
            lhs = rhs; width = 1;
        end
    end
end

for i=1:si(1)
    for j=1:si(2)
        output(i,j) = sr2inv(sr1(input(i,j)+1)+1);
    end
end

