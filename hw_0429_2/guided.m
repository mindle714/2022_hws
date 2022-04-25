function output = guided(input, guide, ksize, eps)
input = im2double(input);
guide = im2double(guide);

output = input;
si = size(input);

radius = fix(ksize/2);
pad_input = padarray(input, [radius, radius]);
pad_guide = padarray(guide, [radius, radius]);

m_guide = guide;
v_guide = guide;

for i=1:si(1)
    for j=1:si(2)
        win_guide = pad_guide(i:i+ksize-1, j:j+ksize-1);
        mk = sum(win_guide, 'all') / (ksize^2);
        vk = sum((win_guide - mk).^2, 'all') / (ksize^2);
        m_guide(i,j) = mk; v_guide(i,j) = vk;
    end
end

a = zeros(size(input));
b = zeros(size(input));

for i=1:si(1)
    for j=1:si(2)
        win_input = pad_input(i:i+ksize-1, j:j+ksize-1);
        win_guide = pad_guide(i:i+ksize-1, j:j+ksize-1);
        m_win_input = sum(win_input, 'all') / (ksize^2);

        ak = sum(win_guide .* win_input - m_guide(i,j) * m_win_input, 'all');
        ak = ak / (ksize^2 * (v_guide(i,j) + eps));
        a(i,j) = ak;

        bk = m_win_input - ak * m_guide(i,j);
        b(i,j) = bk;
    end
end

pad_a = padarray(a, [radius, radius]);
pad_b = padarray(b, [radius, radius]);

for i=1:si(1)
    for j=1:si(2)
        win_ak = pad_a(i:i+ksize-1, j:j+ksize-1);
        m_win_ak = sum(win_ak, 'all') / (ksize^2);

        win_bk = pad_b(i:i+ksize-1, j:j+ksize-1);
        m_win_bk = sum(win_bk, 'all') / (ksize^2);

        output(i,j) = m_win_ak * input(i,j) + m_win_bk;
    end
end