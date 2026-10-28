% Replacement for the nested function gaussianfit of MAIN_CameraBeamAnalyzer.m
% (Curve Fitting Toolbox branch changed; the no-toolbox branch is unchanged and uses the
% corrected Utils/myGaussFit.m)
    function [xfit,yfit,waist]=gaussianfit(x,y,linemask)
        % Gaussian fit of a profile. waist = 1/e2 radius (diameter = 2*waist = 4 sigma).
        % 18/10/2026: - model a1*exp(-((x-b1)/c1)^2)+d1, i.e. gauss1 + a baseline d1 (gauss1 has
        %               none, which overestimated the width of profiles lying on a background)
        %             - excluded points: outside the ROI and first/last two ([1 2 n-1 n])
        %             - start point computed here (a custom model has a random one by default)
        %             - a failed fit gives a warning and no curve instead of an error
        xfit=[]; yfit=[]; waist=0;
        
        if exist('fit','file') >=2 % Requires the curve fitting toolbox
            try
                n = length(x);
                x = x(:); y = y(:);
                exclude = [find(linemask(:)==0)' 1 2 n-1 n find(~isfinite(y))'];
                exclude = unique(exclude(exclude>=1 & exclude<=n));
                y(~isfinite(y)) = 0;
                keep = true(n,1); keep(exclude) = false;
                xk = x(keep); yk = y(keep);
                if numel(xk) < 4 % 4 coefficients
                    error('not enough points left after exclusion');
                end
                % Start point [a b c d]
                ys = sort(yk);
                d0 = mean(ys(1:min(5,numel(ys)))); % baseline: mean of the 5 lowest values
                [ymax,imax] = max(yk);
                a0 = ymax - d0;
                b0 = xk(imax);
                above = find(yk >= d0 + a0/2);
                c0 = abs(xk(above(end)) - xk(above(1)))/(2*sqrt(log(2))); % FWHM -> c
                if ~isfinite(c0) || c0 <= 0
                    c0 = abs(xk(end)-xk(1))/4;
                    if c0 == 0, c0 = 1; end
                end
                ft = fittype('a1*exp(-((x-b1)/c1)^2)+d1','independent','x','coefficients',{'a1','b1','c1','d1'});
                f = fit(x,y,ft,'StartPoint',[a0 b0 c0 d0],'Exclude',exclude);
                coefs=coeffvalues(f); %[a1 b1 c1 d1]
                waist=sqrt(2)*abs(coefs(3));
            catch err
                warning('Gaussian fit failed: %s', err.message);
                return
            end
        else
            [sigma, mu, amp, y0] = myGaussFit(x(linemask==1), y(linemask==1));
            f = @(x) y0 + (amp/(sqrt(2*pi)*sigma))*exp( -(x - mu).^2 ./ (2*sigma.^2));
            waist = 2*sigma;
        end
        
        if waist~=0
            xfitstep=0.1*(x(length(x))-x(1))/(length(x)-1);
            xfit=(x(1):xfitstep:x(length(x)));
            yfit=f(xfit);
        end
    end
