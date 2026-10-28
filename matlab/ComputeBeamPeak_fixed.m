% Replacement for the nested function ComputeBeamPeak of MAIN_CameraBeamAnalyzer.m
% (needs the Image Processing Toolbox: ordfilt2)
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
