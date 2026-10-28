% Exports MATLAB reference values for pycba.core (see tests/test_core.py).
%
% Run it from its own folder (e.g. <project>/Python migration/): Utils/ and SampleBeams/ are looked for in ../Utils and
% ../SampleBeams (export_setup.m, which stops with an error if they are missing). Needs max(...,'linear'), as MAIN.
% It writes core_<name>.mat files in this folder; copy them to
% pyCBA/tests/matlab_reference/ .
%
% The Compute* functions below are COPIES of the nested functions of
% MAIN_CameraBeamAnalyzer.m, unchanged except for one added line,
% "global FILTERS;", needed because they can no longer see the main function's
% variables. If you modify the originals, update the copies.

[~, sampleDir] = export_setup();
if ~exist('knifeEdgeAxis','file')
    error('Utils/knifeEdgeAxis.m (matlab/knifeEdgeAxis.m of pyCBA) is missing: the knife-edge reference would be obsolete.');
end
if ~exist('widthAtLevel','file')
    error('Utils/widthAtLevel.m (matlab/widthAtLevel.m of pyCBA) is missing: the 1/e2 reference would be obsolete.');
end
% remove the .mat of a previous run, so that a stale file (sample renamed, read failure...) cannot survive unnoticed
delete('core_*.mat');
global FILTERS
warning('off','all');

% ---- 1. synthetic images (saved in the .mat, so Python uses the very same pixels)
rng(1);
[xx,yy] = meshgrid(1:160,1:128);
g = @(cx,cy,sx,sy,a) a*exp(-(xx-cx).^2/(2*sx^2)-(yy-cy).^2/(2*sy^2));
cases = struct('name',{},'img',{},'ps',{});
cases(end+1) = struct('name','synth_clean',    'img',g(80.3,60.7,12,7,1000),                          'ps',5.5);
cases(end+1) = struct('name','synth_noisy',    'img',g(70,64,10,10,800)+10*randn(128,160)+50,         'ps',1);
hot = g(70,64,10,10,800)+10*randn(128,160)+50; hot(20,30) = 5000;
cases(end+1) = struct('name','synth_hotpixel', 'img',hot,                                             'ps',5.5);
nn = g(75,60,9,9,500); nn([5 40 90],[10 11 100]) = NaN;
cases(end+1) = struct('name','synth_nan',      'img',nn,                                              'ps',5.5);
[xt,yt] = meshgrid(1:50,1:40);                       % own grid: this image is 40x50
tie = 3*exp(-(xt-25).^2/(2*5^2)-(yt-20).^2/(2*5^2));
tie(30,5) = 9; tie(10,40) = 9;                        % two equal maxima (tie-break test)
cases(end+1) = struct('name','synth_tie',      'img',tie,                                             'ps',1);

% ---- 2. real sample files
files = dir(fullfile(sampleDir,'*')); files = files(~[files.isdir]);
nreal = 0;
for i = 1:numel(files)
    try
        r = readAsciiFromFile(fullfile(sampleDir,files(i).name));
    catch err
        r = [];
        fprintf('  SKIPPED %s: %s\n', files(i).name, err.message);
    end
    if numel(r) > 4
        cases(end+1) = struct('name',files(i).name,'img',r,'ps',5.5); %#ok<SAGROW>
        nreal = nreal + 1;
    else
        fprintf('  SKIPPED %s: not an image (numel <= 4)\n', files(i).name);
    end
end
if nreal == 0
    error('No real sample could be read in %s: aborting (only synthetic cases would be exported).', sampleDir);
end
fprintf('%d real sample(s) read.\n', nreal);

% ---- 3. compute and save
for i = 1:numel(cases)
    img = cases(i).img; pixelsize = cases(i).ps; FILTERS.pixelsize = pixelsize;
    [Xc,Yc]   = ComputeCentroid(img);
    centroid  = [Xc Yc];
    [a,b]     = ComputePeakCenter(img);                peakcenter = [a b];
    [a,b,c,d] = ComputeD4sigma(img,Xc,Yc);             d4sigma    = [a b c d];
    [a,b,c,d] = ComputeDpeak(img);                     dpeak      = [a b c d];
    [a,b,c,d] = ComputeDKnifeEdge(img,Xc,Yc);          dknife     = [a b c d];
    [a,b,c,d] = ComputeDAperture(img,Xc,Yc);           daperture  = [a b c d];
    [a,b,c,d] = ComputeDApertureFixed(img,Xc,Yc);      daperture_fixed = [a b c d];
    peak_smart = ComputeBeamPeak(img);
    peak_smart_old = ComputeBeamPeakOld(img);
    peak_max   = ComputeBeamPeak(img,'max');
    fname = ['core_' regexprep(cases(i).name,'[^\w]','_') '.mat'];
    save(fname,'img','pixelsize','centroid','peakcenter','d4sigma','dpeak','dknife', ...
         'daperture','daperture_fixed','peak_smart','peak_smart_old','peak_max','-v7');
    fprintf('%-40s D4s=[%.4g %.4g]  peak(smart)=%.6g peak(max)=%.6g\n', cases(i).name, d4sigma(3), d4sigma(4), peak_smart, peak_max);
end

% =====================================================================
% COPIES of the nested functions (+ "global FILTERS")
% =====================================================================
    function [Xcentroid, Ycentroid] = ComputeCentroid(image)
        global FILTERS
        % CENTROID
        pixelsize=FILTERS.pixelsize;
        energy=sum(sum(image,'omitnan'),'omitnan')*(pixelsize.^2);
        Xvector=cumsum(ones(1,size(image,2)),'omitnan')*pixelsize; % columns
        Yvector=cumsum(ones(size(image,1),1),'omitnan')*pixelsize; % lines
        Xsum=sum(image,1,'omitnan')*pixelsize;% sum along Y
        Ysum=sum(image,2,'omitnan')*pixelsize;% sum along X
        Xcentroid=(sum(Xsum.*Xvector,'omitnan')*pixelsize)/energy;%centroid
        Ycentroid=(sum(Ysum.*Yvector,'omitnan')*pixelsize)/energy;
    end

    function [Xpeak, Ypeak] = ComputePeakCenter(image)
        global FILTERS
        % CENTER
        [maxincol, maxpos]=max(image,[],1);%row vector of maximum value for each column
        [~, colmax]=max(maxincol);%cmax = column of maximum
        rowmax=maxpos(colmax);%rmax = row of maximum
        Xpeak=colmax*FILTERS.pixelsize;%maximum amplitude
        Ypeak=rowmax*FILTERS.pixelsize;
    end

    function [X0, Y0, DiamH, DiamV] = ComputeD4sigma(image, Xcentroid, Ycentroid)
        global FILTERS
        % 4 SIGMA DIAMETER
        pixelsize=FILTERS.pixelsize;
        energy = sum(sum(image,'omitnan'),'omitnan')*(pixelsize.^2);
        Xvector=cumsum(ones(1,size(image,2)),'omitnan')*pixelsize; % columns
        Yvector=cumsum(ones(size(image,1),1),'omitnan')*pixelsize; % lines
        Xsum=sum(image,1,'omitnan')*pixelsize;% sum along Y
        Ysum=sum(image,2,'omitnan')*pixelsize;% sum along X
        Xsigma=real(sqrt((sum(Xsum.*((Xvector-Xcentroid)).^2,'omitnan')*pixelsize)/energy));
        Ysigma=real(sqrt((sum(Ysum.*((Yvector-Ycentroid)).^2,'omitnan')*pixelsize)/energy));
        DiamH=4*Xsigma;
        DiamV=4*Ysigma;
        X0 = Xcentroid;
        Y0 = Ycentroid;
    end

    function [X0, Y0, DiamH, DiamV] = ComputeDpeak(image, varargin)
        global FILTERS
        % 1/e2 PEAK DIAMETER
        pixelsize=FILTERS.pixelsize;
        [maxincol, maxpos]=max(image,[],1);%row vector of maximum value for each column
        [themax, colmax]=max(maxincol);%cmax = column of maximum
        rowmax=maxpos(colmax);% = row of maximum
        X0=colmax*pixelsize;%maximum amplitude
        Y0=rowmax*pixelsize;
        level=themax/exp(2);
        % SUB-PIXEL since 23/10/2026: needs Utils/widthAtLevel.m, see pyCBA/matlab/DPEAK_SUBPIXEL.txt
        DiamH=widthAtLevel(image(rowmax,:), colmax, level)*pixelsize;
        DiamV=widthAtLevel(image(:,colmax), rowmax, level)*pixelsize;
    end

    function [X0, Y0, DiamH, DiamV] = ComputeDKnifeEdge(image, Xcentroid, Ycentroid)
        global FILTERS
        % Beam diameters, knife-edge method (SUB-PIXEL since 22/10/2026: needs Utils/knifeEdgeAxis.m, see
        % pyCBA/matlab/KNIFE_EDGE_SUBPIXEL.txt)
        pixelsize=FILTERS.pixelsize;
        X0=Xcentroid;
        Y0=Ycentroid;
        KNIFE_LOW  = 0.5*erfc( 1/sqrt(2)); % 0.158655 (-1 sigma of a Gaussian)
        KNIFE_HIGH = 0.5*erfc(-1/sqrt(2)); % 0.841345 (+1 sigma)
        energy = sum(sum(image,'omitnan'),'omitnan')*(pixelsize.^2);
        [xlow, xhigh] = knifeEdgeAxis(sum(image,1,'omitnan')*(pixelsize.^2), energy, KNIFE_LOW, KNIFE_HIGH, pixelsize);
        [ylow, yhigh] = knifeEdgeAxis(sum(image,2,'omitnan')'*(pixelsize.^2), energy, KNIFE_LOW, KNIFE_HIGH, pixelsize);
        DiamH=2*(xhigh-xlow);
        DiamV=2*(yhigh-ylow);
    end

    function [X0, Y0, DiamH, DiamV] = ComputeDAperture(image, Xcentroid, Ycentroid)
        global FILTERS
        % D APERTURE
        pixelsize=FILTERS.pixelsize;
        X0=Xcentroid;
        Y0=Ycentroid;
        Xvector=cumsum(ones(1,size(image,2)),'omitnan')*pixelsize; % columns
        Yvector=cumsum(ones(size(image,1),1),'omitnan')*pixelsize; % lines
        energy = sum(sum(image,'omitnan'),'omitnan')*(pixelsize.^2);
        [nrows, ncols] = size(image);
        distanceToCentroid = sqrt((repmat(Yvector,1,ncols)-Y0).^2 + (repmat(Xvector,nrows,1)-X0).^2);
        
        drmin = pixelsize*sqrt(2);
        dfmin = (1/1000)*energy;
        rb = 0.5*max(nrows,ncols)*pixelsize;
        ra = 1*pixelsize;
        
        for k=1:100
            tempmaska = distanceToCentroid < ra;
            tempmaskb = distanceToCentroid < rb;
            fa = sum(sum(image.*tempmaska),'omitnan')*(pixelsize.^2) - energy*0.8647;
            fb = sum(sum(image.*tempmaskb),'omitnan')*(pixelsize.^2) - energy*0.8647;
            rc = ra - ((ra-rb)/(fa-fb))*fa;
            if sign(fa) == sign(fb)
                ra = rc;
            else
                rb = rc;
            end
            if (abs(fb-fa) <= dfmin)
                break;
            end
            if (abs(rb-ra) <= drmin)
                break;
            end
        end
        if k==100
            warning('Could not calculate diameter. Consider using a different aperture or background substraction.');
        end
        DiamH = 2*ra;
        DiamV = 2*ra;
    end

    function [X0, Y0, DiamH, DiamV] = ComputeDApertureFixed(image, Xcentroid, Ycentroid)
        global FILTERS
        % D APERTURE (fixed: 'omitnan' added to the inner sums)
        pixelsize=FILTERS.pixelsize;
        X0=Xcentroid;
        Y0=Ycentroid;
        Xvector=cumsum(ones(1,size(image,2)),'omitnan')*pixelsize; % columns
        Yvector=cumsum(ones(size(image,1),1),'omitnan')*pixelsize; % lines
        energy = sum(sum(image,'omitnan'),'omitnan')*(pixelsize.^2);
        [nrows, ncols] = size(image);
        distanceToCentroid = sqrt((repmat(Yvector,1,ncols)-Y0).^2 + (repmat(Xvector,nrows,1)-X0).^2);
        
        drmin = pixelsize*sqrt(2);
        dfmin = (1/1000)*energy;
        rb = 0.5*max(nrows,ncols)*pixelsize;
        ra = 1*pixelsize;
        
        for k=1:100
            tempmaska = distanceToCentroid < ra;
            tempmaskb = distanceToCentroid < rb;
            fa = sum(sum(image.*tempmaska,'omitnan'),'omitnan')*(pixelsize.^2) - energy*0.8647;
            fb = sum(sum(image.*tempmaskb,'omitnan'),'omitnan')*(pixelsize.^2) - energy*0.8647;
            rc = ra - ((ra-rb)/(fa-fb))*fa;
            if sign(fa) == sign(fb)
                ra = rc;
            else
                rb = rc;
            end
            if (abs(fb-fa) <= dfmin)
                break;
            end
            if (abs(rb-ra) <= drmin)
                break;
            end
        end
        if k==100
            warning('Could not calculate diameter. Consider using a different aperture or background substraction.');
        end
        DiamH = 2*ra;
        DiamV = 2*ra;
    end

    function peak = ComputeBeamPeak(r,varargin)
        % Find beam maximums avoiding noise errors
        % Fallback (too few pixels near the maximum): removes HOT PIXELS only. The former
        % version replaced every pixel farther than 5.8 MAD from the median, which also
        % erased the beam and returned a noise-level "peak".
        threshold = 5.8;
        smoothStart=0.2/100;
        N=numel(r);
        Nthreshold=max(4,round(N*1e-4));
        r(isinf(r))=0; % Removes Inf
        [themax,maxidx2]=max(r,[],[1 2],'linear','omitnan');
        if abs(nargin)>1
            if strcmpi(varargin{1},'max')
                threshold = Inf;
            elseif strcmpi(varargin{1},'smart')
                %threshold = threshold;
            else
                threshold = varargin{1};
            end
        end
        if ~isfinite(threshold)
            peak=themax;
            return
        else
            smoothEnd=min(10*smoothStart,0.2);
            [X0, Y0] = ind2sub(size(r),maxidx2);
            X=[];Y=[];
            for smooth=smoothStart:smoothStart:smoothEnd
                [X,Y] = find(r>themax*(1-smooth));
                if length(X) >= Nthreshold; break; end
            end
            distances = sqrt((X-X0).^2+(Y-Y0).^2);
            [distances2, idx] = sort(distances);
            steps = diff(distances2);
            valid_idx = idx(steps<=sqrt(2));
            
            if length(valid_idx) < Nthreshold
                % Hot pixels: >= 3x their 3rd brightest neighbour (of 8) AND > threshold MADs above it
                med = median(r,[1 2],'omitnan');
                mdev = median(abs(r-med),[1 2],'omitnan');
                rf = r; rf(isnan(rf)) = -Inf;
                kth = ordfilt2(rf, 6, [1 1 1;1 0 1;1 1 1], 'zeros'); % 3rd brightest neighbour
                hot = (rf > 3*kth) & ((rf-kth) > threshold*mdev);
                r(hot) = kth(hot);
                [themax,maxidx2]=max(r,[],[1 2],'linear','omitnan'); % max of the cleaned image
                [X0, Y0] = ind2sub(size(r),maxidx2);
                for smooth=smoothStart:smoothStart:2*smoothEnd
                    [X,Y] = find(r>themax*(1-smooth));
                    if length(X) >= Nthreshold; break; end
                end 
            end
            distances = sqrt((X-X0).^2+(Y-Y0).^2);
            [distances2, idx] = sort(distances);
            steps = diff(distances2);
            valid_idx = idx(steps<=sqrt(2));
            valid_X = X(valid_idx);
            valid_Y = Y(valid_idx);
            valid_ridx = sub2ind(size(r),valid_X,valid_Y);

            if ~isempty(valid_ridx)
                peak=mean(mean(r(valid_ridx),'omitnan'),'omitnan');
            else
                peak=themax;
            end
        end
    end

    function peak = ComputeBeamPeakOld(r,varargin)
        % Find beam maximums avoiding noise errors
        threshold = 5.8;
        smoothStart=0.2/100;
        N=numel(r);
        Nthreshold=max(4,round(N*1e-4));
        r(isinf(r))=0; % Removes Inf
        [themax,maxidx2]=max(r,[],[1 2],'linear','omitnan');
        if abs(nargin)>1
            if strcmpi(varargin{1},'max')
                threshold = Inf;
            elseif strcmpi(varargin{1},'smart')
                %threshold = threshold;
            else
                threshold = varargin{1};
            end
        end
        if ~isfinite(threshold)
            peak=themax;
            return
        else
            smoothEnd=min(10*smoothStart,0.2);
            [X0, Y0] = ind2sub(size(r),maxidx2);
            X=[];Y=[];
            for smooth=smoothStart:smoothStart:smoothEnd
                [X,Y] = find(r>themax*(1-smooth));
                if length(X) >= Nthreshold; break; end
            end
            distances = sqrt((X-X0).^2+(Y-Y0).^2);
            [distances2, idx] = sort(distances);
            steps = diff(distances2);
            valid_idx = idx(steps<=sqrt(2));
            
            if length(valid_idx) < Nthreshold
                med = median(r,[1 2],'omitnan');
                d = abs(r-med);
                mdev=median(d,[1 2],'omitnan');
                s = abs(d/mdev);
                r(s>threshold)=med;
                [themax2,maxidx2]=max(r,[],[1 2],'linear','omitnan');
                [X0, Y0] = ind2sub(size(r),maxidx2);
                for smooth=smoothStart:smoothStart:2*smoothEnd
                    [X,Y] = find(r>themax2*(1-smooth));
                    if length(X) >= Nthreshold; break; end
                end 
            end
            distances = sqrt((X-X0).^2+(Y-Y0).^2);
            [distances2, idx] = sort(distances);
            steps = diff(distances2);
            valid_idx = idx(steps<=sqrt(2));
            valid_X = X(valid_idx);
            valid_Y = Y(valid_idx);
            valid_ridx = sub2ind(size(r),valid_X,valid_Y);

            if ~isempty(valid_ridx)
                peak=mean(mean(r(valid_ridx),'omitnan'),'omitnan');
            else
                peak=themax;
            end
        end
    end
