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
