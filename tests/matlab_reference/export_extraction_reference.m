% Exports MATLAB reference values for the profile extraction (see tests/test_profiles.py).
%
% Run from any folder (no toolbox needed).
% Writes extract_<name>.mat and l1l2_<name>.mat in the current folder; copy them to
% pyCBA/tests/matlab_reference/ .
% It uses the REAL spdiags and interp2. The code below is a copy of the computation part of
% ProfileButton_Callback and Profile45Button_Callback (no figures, no file saving), of
% lineProfile.m (new ProfileL1L2Button_Callback), of the ORIGINAL smoothOutliers/
% ComputeProfilesMinMax ('_orig') and of the corrected ComputeProfilesMinMax_fixed.m ('_new').

warning('off','all');
rng(5); ps = 5.5;
cases = struct('name',{},'image',{},'roimask',{},'has_roi',{},'xpos',{},'ypos',{});
cases(end+1) = struct('name','tall',   'image',randi(100,9,6),  'roimask',[],'has_roi',0,'xpos',3*ps,    'ypos',5*ps);
cases(end+1) = struct('name','wide',   'image',randi(100,6,9),  'roimask',[],'has_roi',0,'xpos',6*ps,    'ypos',2*ps);
cases(end+1) = struct('name','square', 'image',randi(100,8,8),  'roimask',[],'has_roi',0,'xpos',4*ps,    'ypos',4*ps);
cases(end+1) = struct('name','halfpx', 'image',randi(100,7,10), 'roimask',[],'has_roi',0,'xpos',4.5*ps,  'ypos',3.5*ps);
cases(end+1) = struct('name','corner', 'image',randi(100,7,10), 'roimask',[],'has_roi',0,'xpos',9*ps,    'ypos',1*ps);
[X,Y] = meshgrid(1:60,1:40);
beam = 900*exp(-((X-33).^2/(2*8^2)+(Y-19).^2/(2*5^2)))+40+4*randn(40,60); beam(19,40) = 30000;
ell = ((X-30)/28).^2+((Y-20)/17).^2 < 1;
cases(end+1) = struct('name','roi_beam','image',beam,'roimask',ell,'has_roi',1,'xpos',33*ps,'ypos',19*ps);
nanimg = 900*exp(-((X-33).^2/(2*8^2)+(Y-19).^2/(2*5^2)))+40+4*randn(40,60); nanimg(19,[10 50]) = NaN; nanimg(5,33) = NaN;
cases(end+1) = struct('name','with_nan','image',nanimg,'roimask',[],'has_roi',0,'xpos',33*ps,'ypos',19*ps);

for i = 1:numel(cases)
    c = cases(i); image = c.image; roimask = c.roimask; has_roi = c.has_roi; xpos = c.xpos; ypos = c.ypos; pixelsize = ps; %#ok<NASGU>
    R = hv_profiles(image,has_roi,roimask,xpos,ypos,ps);
    D = diag_profiles(image,has_roi,roimask,xpos,ypos,ps);
    Hvector=R.Hvector; Hprofile=R.Hprofile; Hmask=R.Hmask; Vvector=R.Vvector; Vprofile=R.Vprofile; Vmask=R.Vmask; %#ok<NASGU>
    minusVector=D.minusVector; minusProfile=D.minusProfile; minusMask=D.minusMask; %#ok<NASGU>
    plusVector=D.plusVector; plusProfile=D.plusProfile; plusMask=D.plusMask; %#ok<NASGU>
    [a,b,cc,d] = ComputeProfilesMinMaxOrig(R.Hprofile,R.Hmask,R.Vprofile,R.Vmask); minmax_hv_orig = [a b cc d]; %#ok<NASGU>
    [a,b,cc,d] = ComputeProfilesMinMaxNew(R.Hprofile,R.Hmask,R.Vprofile,R.Vmask);  minmax_hv_new = [a b cc d]; %#ok<NASGU>
    [a,b,cc,d] = ComputeProfilesMinMaxOrig(D.minusProfile,D.minusMask,D.plusProfile,D.plusMask); minmax_diag_orig = [a b cc d]; %#ok<NASGU>
    [a,b,cc,d] = ComputeProfilesMinMaxNew(D.minusProfile,D.minusMask,D.plusProfile,D.plusMask);  minmax_diag_new = [a b cc d]; %#ok<NASGU>
    save(['extract_' c.name '.mat'],'image','roimask','has_roi','xpos','ypos','pixelsize', ...
         'Hvector','Hprofile','Hmask','Vvector','Vprofile','Vmask', ...
         'minusVector','minusProfile','minusMask','plusVector','plusProfile','plusMask', ...
         'minmax_hv_orig','minmax_hv_new','minmax_diag_orig','minmax_diag_new','-v7');
    fprintf('%-10s H=%dx%d diag=%d  minmax_hv_orig=%s  new=%s\n', c.name, numel(R.Hprofile), numel(R.Vprofile), numel(D.minusProfile), mat2str(minmax_hv_orig,4), mat2str(minmax_hv_new,4));
end

% ---- L1/L2 profiles: elongated Gaussian, axes at several angles, + one ROI case
n = 160; c0 = 80.5; [X,Y] = meshgrid(1:n,1:n); pixelsize = ps; %#ok<NASGU>
for ang = [0 10 30 -25 45 60 135 999]
    has_roi = (ang == 999); if has_roi, th = 20*pi/180; else, th = ang*pi/180; end
    u = (X-c0)*cos(th)+(Y-c0)*sin(th); v = -(X-c0)*sin(th)+(Y-c0)*cos(th);
    image = 1000*exp(-u.^2/(2*14^2)-v.^2/(2*6^2)); %#ok<NASGU>
    L = 60;                                                 % end points, 1-based pixels -> length units
    posL1 = ps*[c0-L*cos(th) c0-L*sin(th); c0+L*cos(th) c0+L*sin(th)];
    posL2 = ps*[c0+L*sin(th) c0-L*cos(th); c0-L*sin(th) c0+L*cos(th)]; %#ok<NASGU>
    if has_roi, roimask = (X > 60) & (X < 110); else, roimask = []; end %#ok<NASGU>
    if has_roi, cropped = image.*roimask; else, cropped = image; end
    [L1Vector,L1Profile,L1Mask] = lineProfile(cropped,roimask,posL1,ps); %#ok<ASGLU>
    [L2Vector,L2Profile,L2Mask] = lineProfile(cropped,roimask,posL2,ps); %#ok<ASGLU>
    angle = atan2(posL1(2,2)-posL1(1,2),posL1(2,1)-posL1(1,1))*180/pi; %#ok<NASGU>
    save(sprintf('l1l2_angle%d.mat',ang),'image','roimask','has_roi','posL1','posL2','pixelsize','angle', ...
         'L1Vector','L1Profile','L1Mask','L2Vector','L2Profile','L2Mask','-v7');
    fprintf('l1l2 angle %4d: %d/%d points\n', ang, numel(L1Profile), numel(L2Profile));
end

% =====================================================================
% COPIES of the computation part of the callbacks
% =====================================================================
function R = hv_profiles(rfilt,has_roi,roimask,Xpos,Ypos,pixelsize)
    % ProfileButton_Callback
    dimY = size(rfilt,1); dimX = size(rfilt,2);
    linenum=round(Ypos/pixelsize); colnum=round(Xpos/pixelsize);
    if has_roi
        mask = logical(roimask);                  % createMask returns a logical
        croppedimg=rfilt.*mask;
    else
        croppedimg=rfilt;
        mask = ones(dimY,dimX);
    end
    R.Hvector=transpose((1:size(croppedimg,2))*pixelsize);
    R.Hprofile=transpose(croppedimg(linenum,:));
    R.Hmask=transpose(mask(linenum,:));
    R.Vvector=transpose((1:size(croppedimg,1))*pixelsize);
    R.Vprofile=croppedimg(:,colnum);
    R.Vmask=mask(:,colnum);
end

function D = diag_profiles(rfilt,has_roi,roimask,Xpos,Ypos,pixelsize)
    % Profile45Button_Callback
    dimY = size(rfilt,1); dimX = size(rfilt,2);
    if has_roi
        mask = logical(roimask);
        croppedimg=rfilt.*mask;
    else
        croppedimg=rfilt;
        mask = ones(dimY,dimX);
    end
    linenum=round(Ypos/pixelsize); colnum=round(Xpos/pixelsize);
    minusProfile=spdiags(croppedimg,colnum-linenum);
    plusProfile=spdiags(flipud(croppedimg),colnum-(size(croppedimg,1)-linenum)-1);
    D.minusVector=transpose((1:size(minusProfile))*pixelsize*sqrt(2));
    D.plusVector=transpose((1:size(plusProfile))*pixelsize*sqrt(2));
    D.minusMask=spdiags(mask,colnum-linenum);
    D.plusMask=spdiags(flipud(mask),colnum-(size(mask,1)-linenum)-1);
    D.minusProfile = minusProfile; D.plusProfile = plusProfile;
end

% ---- lineProfile (= matlab/lineProfile.m)
function [vec, prof, mask] = lineProfile(img, roimask, pos, pixelsize)
% Profile of an image along a segment, sampled every pixelsize by bilinear interpolation.
%   pos      = [x1 y1; x2 y2], end points in length units. Pixel (r,c) is at (c*pixelsize, r*pixelsize).
%   roimask  = logical image of the ROI, or [] if there is none.
%   img      = image already cropped by the ROI (rfilt.*roimask).
%   vec      = distance from the first end point (length units), column vector.
%   mask     = 1 for valid points, 0 for points outside the image, outside the ROI or not finite
%              (their value is then set to 0).
% 18/10/2026 - replaces the imrotate-based extraction of ProfileL1L2Button_Callback.
    [nrows, ncols] = size(img);
    p1 = pos(1,:)/pixelsize; p2 = pos(2,:)/pixelsize;      % [col row], 1-based pixels
    len = hypot(p2(1)-p1(1), p2(2)-p1(2));
    n = floor(len + 1e-9) + 1;
    t = (0:n-1)';
    if len > 0
        cols = p1(1) + t*(p2(1)-p1(1))/len;
        rows = p1(2) + t*(p2(2)-p1(2))/len;
    else
        cols = p1(1); rows = p1(2);
    end
    eps0 = 1e-9;
    inside = cols>=1-eps0 & cols<=ncols+eps0 & rows>=1-eps0 & rows<=nrows+eps0;
    cc = min(max(cols,1),ncols); rr = min(max(rows,1),nrows);   % clamp the few points on the border
    prof = interp2(img, cc, rr, 'linear');
    bad = ~inside | ~isfinite(prof);
    prof(bad) = 0;
    mask = ~bad;
    if ~isempty(roimask)
        mask = mask & (interp2(double(roimask), cc, rr, 'nearest') > 0.5);
    end
    mask = double(mask);
    vec = t*pixelsize;
end

% ---- ORIGINAL smoothOutliers + ComputeProfilesMinMax
function yout = smoothOutliersOrig(y,varargin)
    if abs(nargin)>=2, threshold=varargin{1}; else, threshold=5; end
    y2 = y(isfinite(y));
    med = median(y2,'omitnan');
    d = abs(y2-med);
    mdev=median(d,'omitnan');
    if mdev~=0
        s = abs(d/mdev);
        y(s>threshold)=med;
    end
    yout=y;
end

function [Hmin, Hmax, Vmin, Vmax] = ComputeProfilesMinMaxOrig(Hprofile, Hmask, Vprofile, Vmask)
    smooth = 2;
    Hprofile2 = Hprofile((Hmask==1)); Hprofile2(isinf(Hprofile2))=0;
    Hprofile2 = smoothOutliersOrig(Hprofile2,5); Hprofile2=sort(Hprofile2); nH = length(Hprofile2);
    Hmin = mean(Hprofile2(1:min(smooth,nH)),'omitnan'); Hmax = mean(Hprofile2(max(nH-smooth,1):end),'omitnan');
    Vprofile2 = Vprofile((Vmask==1)); Vprofile2(isinf(Vprofile2))=0;
    Vprofile2 = smoothOutliersOrig(Vprofile2,5); Vprofile2=sort(Vprofile2); nV = length(Vprofile2);
    Vmin = mean(Vprofile2(1:min(smooth,nV)),'omitnan'); Vmax = mean(Vprofile2(max(nV-smooth,1):end),'omitnan');
end

% ---- CORRECTED ComputeProfilesMinMax (= ComputeProfilesMinMax_fixed.m)
function [Hmin, Hmax, Vmin, Vmax] = ComputeProfilesMinMaxNew(Hprofile, Hmask, Vprofile, Vmask)
    smooth = 2;
    Hprofile2 = Hprofile((Hmask==1)); Hprofile2(isinf(Hprofile2))=0;
    Hprofile2 = removeProfileOutliers(Hprofile2,5); Hprofile2=sort(Hprofile2); nH = length(Hprofile2);
    Hmin = mean(Hprofile2(1:min(smooth,nH)),'omitnan'); Hmax = mean(Hprofile2(max(nH-smooth+1,1):end),'omitnan');
    Vprofile2 = Vprofile((Vmask==1)); Vprofile2(isinf(Vprofile2))=0;
    Vprofile2 = removeProfileOutliers(Vprofile2,5); Vprofile2=sort(Vprofile2); nV = length(Vprofile2);
    Vmin = mean(Vprofile2(1:min(smooth,nV)),'omitnan'); Vmax = mean(Vprofile2(max(nV-smooth+1,1):end),'omitnan');
end

function y = removeProfileOutliers(y, threshold)
    y = y(:);
    fin = isfinite(y);
    if ~any(fin), return; end
    med = median(y(fin));
    mdev = median(abs(y(fin)-med));
    n = numel(y);
    if n >= 3
        left = [y(2); y(1:n-1)];
        right = [y(2:n); y(n-1)];
        k = min(left,right);
        hot = fin & (y > 3*k) & ((y-k) > threshold*mdev);
        y(hot) = k(hot);
    end
    if mdev ~= 0
        y(fin & (med-y > threshold*mdev)) = med;
    end
end
