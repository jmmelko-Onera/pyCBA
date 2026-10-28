% Automatic background estimate (new function, Utils/estimateBackground.m). BASE MATLAB, no toolbox.
% Same algorithm as pycba.core.background.estimate_background.
%
% bkg        : median of a frame 5 % wide along the 4 edges of the image (NaN = dead pixels are ignored).
%              The median ignores isolated hot/dead pixels and a damaged area covering up to ~1/4 of the frame.
% nonUniform : the medians of the 4 bands disagree (> 5 % of peak-bkg): damaged area, gradient...
% high       : bkg > 10 % of the peak: probably the wings of a wide beam on the whole border
% pctPeak    : 100*bkg/peak
% Error if the image is smaller than 20 x 20 or if the frame holds no valid pixel.
function [bkg,nonUniform,high,pctPeak] = estimateBackground(img)
    FRAME = 0.05; MINSIDE = 20; NONUNIFORM = 0.05; HIGH = 0.10;
    [ny,nx] = size(img);
    kr = max(1,ceil(FRAME*ny)); kc = max(1,ceil(FRAME*nx));
    if min(ny,nx) < MINSIDE || 2*kr >= ny || 2*kc >= nx
        error('image too small to estimate the background on its border (minimum %d x %d pixels)',MINSIDE,MINSIDE);
    end
    mask = false(ny,nx);
    mask(1:kr,:) = true; mask(end-kr+1:end,:) = true;
    mask(:,1:kc) = true; mask(:,end-kc+1:end) = true;
    vals = img(mask); vals = vals(isfinite(vals));
    if isempty(vals)
        error('no valid pixel on the image border');
    end
    bkg = median(vals);
    sides = [median(reshape(img(1:kr,:),[],1),'omitnan'), median(reshape(img(end-kr+1:end,:),[],1),'omitnan'), ...
             median(reshape(img(:,1:kc),[],1),'omitnan'), median(reshape(img(:,end-kc+1:end),[],1),'omitnan')];
    peak = max(img(isfinite(img)));
    signal = peak - bkg;
    nonUniform = signal > 0 && max(abs(sides - bkg)) > NONUNIFORM*signal; % NaN side: max ignores it
    high = peak > 0 && bkg > HIGH*peak;
    pctPeak = 100*bkg/peak;
end
