% Replacement for the nested function ComputeProfilesMinMax of MAIN_CameraBeamAnalyzer.m
% (and its helper). Also corrected: the maximum was the mean of the 3 highest values
% (Hprofile2(max(nH-smooth+1,1):end) has smooth+1 elements) while the minimum is the mean of the 2 lowest:
% both use 'smooth' = 2 points now (max(nH-smooth+1,1)).
    function [Hmin, Hmax, Vmin, Vmax] = ComputeProfilesMinMax(Hprofile, Hmask, Vprofile, Vmask)
        % Computes min and max of two profiles
        % 18/10/2026: smoothOutliers(.,5) replaced EVERY point more than 5 MADs from the median,
        % the beam included, so the reported maximum was only 6 to 22 % of the true one unless the
        % beam filled most of the profile. Only HOT points are removed on the high side now.
        smooth = 2;
        Hprofile2 = Hprofile((Hmask==1));
        Hprofile2(isinf(Hprofile2))=0;
        Hprofile2 = removeProfileOutliers(Hprofile2,5);
        Hprofile2=sort(Hprofile2);
        nH = length(Hprofile2);
        Hmin = mean(Hprofile2(1:min(smooth,nH)),'omitnan');
        Hmax = mean(Hprofile2(max(nH-smooth+1,1):end),'omitnan');
        Vprofile2 = Vprofile((Vmask==1));
        Vprofile2(isinf(Vprofile2))=0;
        Vprofile2 = removeProfileOutliers(Vprofile2,5);
        Vprofile2=sort(Vprofile2);
        nV = length(Vprofile2);
        Vmin = mean(Vprofile2(1:min(smooth,nV)),'omitnan');
        Vmax = mean(Vprofile2(max(nV-smooth+1,1):end),'omitnan');
    end

    function y = removeProfileOutliers(y, threshold)
        % High side: hot points = >= 3x their lesser neighbour AND more than 'threshold' MADs above
        % it (isolated hot points and hot pairs). Low side: more than 'threshold' MADs below the
        % median -> median (as before).
        y = y(:);
        fin = isfinite(y);
        if ~any(fin), return; end
        med = median(y(fin));
        mdev = median(abs(y(fin)-med));
        n = numel(y);
        if n >= 3
            left = [y(2); y(1:n-1)];   % end points: their only neighbour
            right = [y(2:n); y(n-1)];
            k = min(left,right);       % lesser neighbour (NaN ignored)
            hot = fin & (y > 3*k) & ((y-k) > threshold*mdev);
            y(hot) = k(hot);
        end
        if mdev ~= 0
            y(fin & (med-y > threshold*mdev)) = med;
        end
    end
