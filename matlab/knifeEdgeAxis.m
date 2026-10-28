% One axis of the knife-edge diameter (Utils/knifeEdgeAxis.m), SUB-PIXEL (22/10/2026). Base MATLAB.
% sums : energy of each column (or line) as a ROW vector, already multiplied by pixelsize^2
% Pixel i spans [i-0.5, i+0.5]*pixelsize and cum(i) is the energy up to its right edge, so the threshold is crossed at
%   (i - 0.5 + (target - cum(i-1))/(cum(i) - cum(i-1))) * pixelsize     (cum(0) = 0)
% The 84 % search starts at the same pixel as the 16 % one (a single pixel may hold both thresholds).
% Returns low=high=0 if the low threshold is never reached, high=0 if only the low one is (as the old function did).
function [low, high] = knifeEdgeAxis(sums, energy, kLow, kHigh, pixelsize)
    low = 0; high = 0;
    cum = cumsum(sums);
    i = find(cum > energy*kLow, 1);
    if isempty(i), return; end
    low = (i - 0.5 + fraction(energy*kLow, previous(cum,i), cum(i)))*pixelsize;
    j = i - 1 + find(cum(i:end) > energy*kHigh, 1);
    if isempty(j), return; end
    high = (j - 0.5 + fraction(energy*kHigh, previous(cum,j), cum(j)))*pixelsize;
end

% Position of the target inside the pixel, clamped to [0,1] (only matters for an image with a negative total energy)
function f = fraction(target, before, after)
    f = min(max((target - before)/(after - before), 0), 1);
end

function p = previous(cum, k)
    if k > 1, p = cum(k-1); else, p = 0; end
end
