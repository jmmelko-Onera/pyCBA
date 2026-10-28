% Utils/widthAtLevel.m -- width (in PIXELS) of the peak of a 1-D profile at a given level, SUB-PIXEL (23/10/2026). Base MATLAB.
% p     : profile (row or column vector), ic : index of the peak pixel, level : e.g. max/exp(2)
% On each side the profile is followed from the peak until the first pixel that is not above 'level', and the crossing
% is interpolated linearly between that pixel and the previous one; the width is the distance between the two crossings.
% A side that never goes down to 'level' (beam cut by the image edge) or that meets a NaN is missing: the width is then
% twice the distance from the peak to the other crossing (old symmetric assumption); 0 if both are missing or if the peak
% is not positive (no 1/e2 level).
function w = widthAtLevel(p, ic, level)
    p = p(:)';  n = numel(p);
    w = 0;
    if ~(p(ic) > 0), return; end
    i = ic - 1;
    while i >= 1 && p(i) > level, i = i - 1; end
    j = ic + 1;
    while j <= n && p(j) > level, j = j + 1; end
    left = NaN;  right = NaN;
    if i >= 1 && isfinite(p(i)), left  = i + (level - p(i))/(p(i+1) - p(i)); end
    if j <= n && isfinite(p(j)), right = (j - 1) + (p(j-1) - level)/(p(j-1) - p(j)); end
    if isnan(left) && isnan(right)
        w = 0;
    elseif isnan(left)
        w = 2*(right - ic);
    elseif isnan(right)
        w = 2*(ic - left);
    else
        w = right - left;
    end
end
