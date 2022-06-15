clear;

datadir     = '../data';    %the directory containing the images
resultsdir  = '../results'; %the directory for dumping results

%parameters
sigma     = 2;
threshold = 0.03;
rhoRes    = 2;
thetaRes  = pi/90;
nLines    = 50;
%end of parameters

imglist = dir(sprintf('%s/*.jpg', datadir));

imgidx = 1;
f = figure();
for i = 1:3%numel(imglist)
    
    %read in images%
    [path, imgname, dummy] = fileparts(imglist(i).name);
    img = imread(sprintf('%s/%s', datadir, imglist(i).name));
    
    if (ndims(img) == 3)
        img = rgb2gray(img);
    end
    
    img = double(img) / 255;
   
    %actual Hough line code function calls%  
    [Im] = myEdgeFilter(img, sigma);   

    subplot(3,2,imgidx);
    imshow(Im>threshold);

    subplot(3,2,imgidx+1);
    [H,rhoScale,thetaScale] = myHoughTransform(Im, threshold, 2, deg2rad(2));
    Ht = H';
    imshow(Ht/max(Ht,[],'all'),[],'XData',rad2deg(thetaScale),'YData',rhoScale,...
            'InitialMagnification','fit');
    xlabel('\theta'), ylabel('\rho');
    axis on, axis normal, hold on;
    imgidx = imgidx + 2;
end
x_width=100 ;y_width=100;
 set(f, 'PaperPosition', [0 0 x_width y_width]);
saveas(f,'ttt.png')    
