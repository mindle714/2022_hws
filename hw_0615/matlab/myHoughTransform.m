function [rho_cnt, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)
thres_Im = (Im > threshold);
[thres_Im_i, thres_Im_j, ~] = find(thres_Im);

thetaScale = -pi/2:thetaRes:pi/2-thetaRes;
[i_x, i_y] = meshgrid(thetaScale, thres_Im_i);
[j_x, j_y] = meshgrid(thetaScale, thres_Im_j);

%rho = (i_y-1) .* cos(i_x) + (j_y-1) .* sin(j_x);
rho = (j_y-1) .* cos(i_x) + (i_y-1) .* sin(j_x);

D = sqrt((size(Im, 1) - 1)^2 + (size(Im, 2) - 1)^2);
diagonal = rhoRes * ceil(D / rhoRes);
rhoScale = -diagonal:rhoRes:diagonal;
%max_rho = max(rho, [], 'all');
%min_rho = min(rho, [], 'all');
%rhoScale = min_rho:rhoRes:max_rho;

si = size(rho);
rho_cnt = zeros(size(thetaScale, 2), size(rhoScale, 2));

for i=1:si(1)
    for j=1:si(2)
        val = rho(i, j);
        [~, midx] = min(abs(val - rhoScale)); 
        rho_cnt(j, midx) = rho_cnt(j, midx) + 1;
    end
end