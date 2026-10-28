% Exports MATLAB reference values for the image pipeline (see tests/test_imagepipeline.py).
%
% Needs the Image Processing Toolbox (imboxfilt). Writes pipeline_<name>.mat in the current
% folder; copy it to pyCBA/tests/matlab_reference/ .
% Contents: imboxfilt with every 'Padding' option AND its default, on a clean image and on one with
% NaN; copies of computeFlicker and of the computation part of update_image (corrected order,
% normalisation method 1); a copy of the scaling part of plot_image (no drawing) using the corrected ComputeBeamPeak.

warning('off','all');
rng(11);
image = 1000*rand(25,40);
image_nan = image; image_nan(10,20) = NaN; image_nan(3,3) = NaN; %#ok<NASGU>
out = struct();
for order = [1 2 3 5 6]
    k = floor(order/2)*2+1;
    out.(sprintf('box_o%d_zeros',order))     = imboxfilt(image,k,'Padding',0);
    out.(sprintf('box_o%d_replicate',order)) = imboxfilt(image,k,'Padding','replicate');
    out.(sprintf('box_o%d_symmetric',order)) = imboxfilt(image,k,'Padding','symmetric');
    out.(sprintf('box_o%d_circular',order))  = imboxfilt(image,k,'Padding','circular');
end
box_o3_default = imboxfilt(image,3); %#ok<NASGU>
box_o5_default = imboxfilt(image,5); %#ok<NASGU>
box_nan_o3_default = imboxfilt(image_nan,3); %#ok<NASGU>

% ---- computeFlicker and update_image, method 1 (copies of the computation parts, CORRECTED order)
raw = 100+50*rand(30,45);
roimask = false(30,45); roimask(:,5:40) = true;               % ROI that does not touch the edge columns... partly
flicker = computeFlickerCopy(raw,[]);
flicker_roi = computeFlickerCopy(raw,roimask); %#ok<NASGU>
deadmask = ones(30,45); deadmask(5,7) = 0; deadmask(20,30) = 0; %#ok<NASGU>
bkg = 12.5; energy = 1.7; filtorder = 4; %#ok<NASGU>
rfilt = raw;
rfilt = rfilt - flicker;
rfilt = rfilt - bkg;
rfilt = rfilt.*deadmask;                                       % dead pixels AFTER the background (they end at 0)
rfilt = rfilt*energy;
if ~isempty(filtorder), rfilt = imboxfilt(rfilt,floor(filtorder/2)*2+1); end
rfilt_ref = rfilt; %#ok<NASGU>

% ---- display limits (copy of the scaling part of plot_image)
[X,Y] = meshgrid(1:60,1:40);
disp_image = 900*exp(-((X-33).^2/(2*8^2)+(Y-19).^2/(2*5^2)))+40-60+4*randn(40,60); disp_image(7,50) = 30000; %#ok<NASGU>
disp_params = [1 1 1 4095 0 0;  1 1 0 4095 0 0;  1 1 1 4095 0 256;  1 3 1 4095 0 0;  1 4 1 4095 0 64; ...
               2 3 1 4095 0 0;  1 1 1 4095 1 0;  1 1 1 4095 1 100;  1 4 1 4095 1 0;  1 4 1 4095 1 64];
disp_lim = zeros(size(disp_params,1),2); disp_check = zeros(size(disp_params,1),1);
for i = 1:size(disp_params,1)
    p = disp_params(i,:);
    [data,vmin,vmax] = display_limits(disp_image,p(1),p(2),p(3),p(4),p(5),p(6));
    disp_lim(i,:) = [vmin vmax]; disp_check(i) = sum(data(isfinite(data)));
end
fn = fieldnames(out);                         % box_o*_* matrices -> plain variables
for i = 1:numel(fn), eval([fn{i} ' = out.(fn{i});']); end
save('pipeline_all.mat','image','image_nan','box_o3_default','box_o5_default','box_nan_o3_default','raw','flicker','deadmask','bkg', ...
     'energy','filtorder','rfilt_ref','roimask','flicker_roi','disp_image','disp_params','disp_lim','disp_check',fn{:},'-v7');
fprintf('Saved pipeline_all.mat\n');
fprintf('imboxfilt default == Padding 0 (zeros)?      %d\n', isequal(box_o3_default, box_o3_zeros));
fprintf('imboxfilt default == Padding replicate?      %d\n', isequal(box_o3_default, box_o3_replicate));
fprintf('NaN image: finite pixels after imboxfilt:    %d of %d\n', nnz(isfinite(box_nan_o3_default)), numel(box_nan_o3_default));

% =====================================================================
% COPY of computeFlicker (the ROI test of the original is decided by the caller: roimask or [])
% =====================================================================
function flicker = computeFlickerCopy(image, roimask)
    if ~isempty(roimask)
        image = image.*roimask;
    end
    [~, cols] = size(image);
    slices = horzcat(1:ceil(0.05*cols), cols-ceil(0.05*cols)+1:cols);
    rowmeans = mean(image(:,slices),2);
    flicker = repmat(rowmeans,1,cols);
end

% =====================================================================
% COPY of the scaling part of plot_image
% =====================================================================
function [data,vmin,vmax] = display_limits(rfilt,imageNum,method,Znormalize,dimZ,logflag,cmaplen)
    customcolormap = zeros(cmaplen,3); if cmaplen == 0, customcolormap = []; end
    if Znormalize==true
        if imageNum==1 && method==3
            Zmax=ComputeBeamPeak(rfilt,'smart');
        elseif imageNum==1 && method==4
            Zmax=ComputeBeamPeak(rfilt,3.5);
        else
            Zmax=ComputeBeamPeak(rfilt,'max');
        end
    else
        Zmax=dimZ;
    end
    if logflag==true
        if Zmax==0; Zmax=1e-9; end
        if method==4 && imageNum==1
            data = sign(rfilt).*(log10(1+abs(rfilt)/(10^(0))));
            symlogScalemin=-log10(1+Zmax); symlogScalemax=log10(1+Zmax);
            if ~isempty(customcolormap)
                symlogScalemax=symlogScalemax+2*(symlogScalemax)/length(customcolormap);
                symlogScalemin=-symlogScalemax;
            end
            vmin = symlogScalemin; vmax = symlogScalemax;
        else
            data = log10(max(rfilt,0));
            logScalemin=log10(Zmax/10000); logScalemax=log10(Zmax);
            if ~isempty(customcolormap)
                logScalemax=logScalemax+2*(logScalemax-logScalemin)/length(customcolormap);
            end
            vmin = logScalemin; vmax = logScalemax;
        end
    else
        data = rfilt;
        if ~isempty(customcolormap)
            rescale=length(customcolormap)/(length(customcolormap)-1);
            if method == 4 && imageNum==1
                scalemax = Zmax*rescale+1e-9;
                scalemin=-scalemax;
            else
                scalemin=-0.5*Zmax/length(customcolormap);
                scalemax=Zmax*rescale-scalemin+1e-9;
            end
        else
            scalemin=0;
            scalemax=Zmax+1e-9;
        end
        lim = sort([scalemin scalemax]); vmin = lim(1); vmax = lim(2);
    end
end

% ---- corrected ComputeBeamPeak (= matlab/ComputeBeamPeak_fixed.m)
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
