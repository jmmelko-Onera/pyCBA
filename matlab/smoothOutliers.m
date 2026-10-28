function yout = smoothOutliers(y,varargin)
% Replaces by the median the finite values farther than 'threshold' MADs from the median.
% 18/10/2026: the outlier test was computed on the finite values only (y2) and applied to the
%             whole vector, which mis-aligned the replaced elements as soon as y contained a
%             NaN or an Inf. It is now applied to the right elements.
    if abs(nargin)>=2
        threshold=varargin{1};
    else
        threshold=5;
    end
    fin = isfinite(y);
    med = median(y(fin),'omitnan');
    d = abs(y-med);
    mdev=median(d(fin),'omitnan');
    if mdev~=0
        s = d/mdev;
        y(fin & s>threshold)=med;
    end
    yout=y;
end
