% ISO 11146-1 second-moment diameters along the PRINCIPAL axes (new function, Utils/computePrincipalAxes.m).
% Base MATLAB (R2016b+: implicit expansion is not even needed, only matrix products are used).
% Same algorithm as pycba.core.compute_principal_axes.
%
%   image      : (possibly masked) image, NaN/Inf ignored
%   Xc, Yc     : centroid, in length units (pixel i is at i*pixelsize)
%   Dmajor, Dminor : 4*sqrt(eigenvalues of the covariance matrix [sxx sxy; sxy syy])
%   angleDeg   : angle of the major axis with the X axis, COUNTER-CLOCKWISE AS SEEN ON SCREEN (image displayed with
%                y downwards), in (-90,90]. Arbitrary (0) for a round beam.
%   ellipticity: Dminor/Dmajor (ISO 11146: > 0.87 = "circular")
%   Dx, Dy     : 4*sigma along X and Y (equal to ComputeD4sigma's DiamH and DiamV)
function [Dmajor,Dminor,angleDeg,ellipticity,Dx,Dy] = computePrincipalAxes(image,Xc,Yc,pixelsize)
    w = image; w(~isfinite(w)) = 0;
    [ny,nx] = size(w);
    dx = (1:nx)*pixelsize - Xc;     % 1 x nx
    dy = (1:ny)*pixelsize - Yc;     % 1 x ny
    total = sum(w(:));
    sxx = (sum(w,1)*(dx.^2)')/total;
    syy = (dy.^2*sum(w,2))/total;
    sxy = (dy*w*dx')/total;
    half = (sxx+syy)/2;
    rad = hypot((sxx-syy)/2, sxy);
    Dmajor = 4*sqrt(max(half+rad,0));
    Dminor = 4*sqrt(max(half-rad,0));
    phi = 0.5*atan2(2*sxy, sxx-syy);          % major axis, in image coordinates (y downwards)
    angleDeg = -phi*180/pi;
    if angleDeg <= -90, angleDeg = angleDeg + 180; end
    ellipticity = Dminor/Dmajor;
    Dx = 4*sqrt(max(sxx,0));
    Dy = 4*sqrt(max(syy,0));
end
