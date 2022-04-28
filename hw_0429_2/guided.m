function output = guided(input, guide, ksize, eps)
si = size(input);
output = zeros(si);

radius = fix(ksize/2);
pad_input = padarray(input, [radius, radius], 'symmetric');
pad_guide = padarray(guide, [radius, radius], 'symmetric');

m_guide = guide;
v_guide = guide;

for i=1:si(1)
    for j=1:si(2)
        for k=1:si(3)
            win_guide = pad_guide(i:i+ksize-1, j:j+ksize-1, k);
            mk = mean(win_guide, 'all');
            vk = mean((win_guide - mk).^2, 'all');
            m_guide(i,j,k) = mk; v_guide(i,j,k) = vk;
        end
    end
end

a = zeros(size(input));
b = zeros(size(input));

for i=1:si(1)
    for j=1:si(2)
        for k=1:si(3)
            win_input = pad_input(i:i+ksize-1, j:j+ksize-1, k);
            win_guide = pad_guide(i:i+ksize-1, j:j+ksize-1, k);
            m_win_input = mean(win_input, 'all');

            ak = mean(win_guide .* win_input - m_guide(i,j,k) * m_win_input, 'all');
            ak = ak / (v_guide(i,j,k) + eps);
            a(i,j,k) = ak;

            bk = m_win_input - ak * m_guide(i,j,k);
            b(i,j,k) = bk;
        end
    end
end

pad_a = padarray(a, [radius, radius], 'symmetric');
pad_b = padarray(b, [radius, radius], 'symmetric');

for i=1:si(1)
    for j=1:si(2)
        for k=1:si(3)
            win_ak = pad_a(i:i+ksize-1, j:j+ksize-1, k);
            win_bk = pad_b(i:i+ksize-1, j:j+ksize-1, k);
            output(i,j,k) = mean(win_ak .* guide(i,j,k) + win_bk, 'all');
        end
    end
end