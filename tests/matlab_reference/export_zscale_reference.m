% Exports MATLAB reference values for normalizeImage, clean_image and the choice of the Z scale
% (see tests/test_zscale.py). Writes zscale_all.mat in the current folder; copy it to
% pyCBA/tests/matlab_reference/ .
%
% Zscales = list used by CameraBeamAnalyzer (menu entries after 'Normalize'). The corrected rule is exported
% (first entry >= 2^n); dimz_orig / menu_orig hold the original 'closest entry' rule for comparison.
% ComputeBeamPeak (the corrected one is fine) must be on the path.

warning('off','all');
Zscales = [1,256,512,1024,2048,4096,8192,16384,32768,65536,131072];

rng(5);
out = struct();
cases = {};
a = 10+5*rand(6,7); b = 4+rand(6,7); c = 1+rand(6,7);
b(2,3) = 0; c(4,4) = -2; b(5,5) = -3;                         % zero / negative denominators
k = 0;
for method = 1:4
    out.(sprintf('n%d_method',k)) = method;
    out.(sprintf('n%d_a',k)) = a; out.(sprintf('n%d_b',k)) = b; out.(sprintf('n%d_c',k)) = c;
    out.(sprintf('n%d_out',k)) = normalizeImageCopy(method,a,b,c);
    k = k+1;
end
out.n_norm = k;

clean_in = rand(5,6); clean_in(2,2) = NaN; clean_in(3,4) = Inf; clean_in(1,6) = -Inf; clean_in(4,1) = -7;
out.clean_in = clean_in;
out.clean_out = clean_imageCopy(clean_in);

rmax = [0.3 0.6 1 2.5 100 254.9 255 256 1000 2000 4095 4096 5000 20000 60000 70000 1e6];
dimz = zeros(size(rmax)); menu_value = zeros(size(rmax)); dimz_orig = dimz; menu_orig = dimz;
for i = 1:numel(rmax)
    [dimz(i), menu_value(i)] = zscaleCopy(rmax(i),Zscales,true);
    [dimz_orig(i), menu_orig(i)] = zscaleCopy(rmax(i),Zscales,false);
end
out.zscales = Zscales; out.rmax = rmax; out.dimz = dimz; out.menu_value = menu_value; out.dimz_orig = dimz_orig; out.menu_orig = menu_orig;
save('zscale_all.mat','-struct','out');
disp('zscale_all.mat written');

function image=normalizeImageCopy(methodNum,imageA,imageB,imageC)
    if methodNum == 1; image=imageA; return; end
    switch methodNum
        case 2, image=imageA-imageB;
        case 3, image=imageA./imageB;
        case 4
            denom = imageB.*imageC;
            image=(imageA-imageB-imageC)./(4*real(sqrt(denom)));
            image(denom<0) = 0;
    end
end

function image2=clean_imageCopy(image)
    image2=image; idx=isfinite(image2); themin=min(min(image2(idx))); image2(~idx)=themin;
end

function [dimZ, menuValue] = zscaleCopy(rmax,Zscales,corrected)
    % copy of the rmax>0 branch of reinitZScale
    maxbinary=bitget(max(0,round(rmax)),32:-1:1);
    maxbinary=maxbinary(find(maxbinary,1,'first'):length(maxbinary));
    ceilbinary=zeros(1,length(maxbinary)+1);
    ceilbinary(1)=1;
    dimZ=max(bin2dec(num2str(ceilbinary)));
    if corrected
        idx = find(Zscales>=dimZ,1,'first');
        if isempty(idx); idx = numel(Zscales); end
    else
        [~,idx]=sort(abs(Zscales-dimZ)); idx = idx(1);
    end
    dimZ=Zscales(idx);
    menuValue=idx+1;
end
