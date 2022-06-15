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
    [H,rhoScale,thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes);

    subplot(3,2,imgidx+1);
    [rhos, thetas] = myHoughLines(H, nLines);
    lines = houghlines(Im>threshold, 180*(thetaScale/pi), rhoScale, [rhos,thetas],'FillGap',5,'MinLength',30);
    
    img2 = img;
    for j=1:numel(lines)
       img2 = drawLine(img2, lines(j).point1, lines(j).point2); 
    end     
    imshow(img2);

    subplot(3,2,imgidx);
    [rhos, thetas] = myHoughLines_wonms(H, nLines);
    lines = houghlines(Im>threshold, 180*(thetaScale/pi), rhoScale, [rhos,thetas],'FillGap',5,'MinLength',30);
    
    img2 = img;
    for j=1:numel(lines)
       img2 = drawLine(img2, lines(j).point1, lines(j).point2); 
    end     
    imshow(img2);

    imgidx = imgidx+2;
end
x_width=100 ;y_width=100;
set(f, 'PaperPosition', [0 0 x_width y_width]);
saveas(f,'ttt.png')