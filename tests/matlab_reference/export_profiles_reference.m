% Exports MATLAB reference values for pycba.core.profiles (see tests/test_profiles.py).
%
% Run it from its own folder: Utils/, SampleBeams/ (and shim/ without toolbox) are looked for in ../ (export_setup.m,
% error if missing). Needs the Curve Fitting Toolbox
% for the 'gauss1' branch. Writes profile_<name>.mat in the current folder; copy them to
% pyCBA/tests/matlab_reference/ .
%
% BEFORE RUNNING: put the corrected myGaussFit.m (pyCBA/matlab/myGaussFit.m) in Utils/.
% gaussianfit is copied below TWICE, with the branch hard-wired (toolbox / no toolbox),
% because the original picks the branch with exist('fit','file'). The toolbox copy is
% matlab/gaussianfit_fixed.m, without the try/catch (a failure is flagged in the file).

[~, sampleDir, ~, shimsDir] = export_setup();
if ~license('test','Curve_Fitting_Toolbox') || isempty(which('fittype'))
    if ~isfolder(shimsDir)
        error('No Curve Fitting Toolbox and shim folder not found: %s', shimsDir);
    end
    addpath(shimsDir);
    fprintf('No Curve Fitting Toolbox: fake fittype from %s\n', shimsDir);
end
delete('profile_*.mat');   % stale files of a previous run must not survive
warning('off','all');
rng(2);
n = 120; x = (1:n)'*5.5;
cases = struct('name',{},'x',{},'y',{},'mask',{});
cases(end+1) = struct('name','synth_clean',  'x',x,'y',800*exp(-((x-300)/80).^2),                        'mask',ones(n,1));
cases(end+1) = struct('name','synth_noisy',  'x',x,'y',800*exp(-((x-250)/60).^2)+30+8*randn(n,1),         'mask',ones(n,1));
yy = 800*exp(-((x-300)/80).^2)+3*randn(n,1); yy(x>450) = 500;
cases(end+1) = struct('name','synth_masked', 'x',x,'y',yy,                                                'mask',double(x<=450));
cases(end+1) = struct('name','synth_rightedge','x',x,'y',800*exp(-((x-640)/80).^2)+3*randn(n,1),         'mask',ones(n,1));
cases(end+1) = struct('name','synth_leftedge', 'x',x,'y',800*exp(-((x-25)/80).^2)+3*randn(n,1),          'mask',ones(n,1));

% real profiles: row and column through the brightest pixel of each sample file
files = dir(fullfile(sampleDir,'*')); files = files(~[files.isdir]);
nreal = 0;
for i = 1:numel(files)
    try
        r = readAsciiFromFile(fullfile(sampleDir,files(i).name));
    catch err
        r = []; fprintf('  SKIPPED %s: %s\n', files(i).name, err.message);
    end
    if size(r,1) < 20 || size(r,2) < 20, continue; end
    nreal = nreal + 1;
    [~,imax] = max(r(:)); [rr,cc] = ind2sub(size(r),imax);
    nm = regexprep(files(i).name,'[^\w]','_');
    cases(end+1) = struct('name',[nm '_H'],'x',(1:size(r,2))'*5.5,'y',r(rr,:)','mask',ones(size(r,2),1)); %#ok<SAGROW>
    cases(end+1) = struct('name',[nm '_V'],'x',(1:size(r,1))'*5.5,'y',r(:,cc),'mask',ones(size(r,1),1)); %#ok<SAGROW>
end
if nreal == 0
    error('No real sample could be read in %s: aborting (only synthetic cases would be exported).', sampleDir);
end
fprintf('%d real sample(s) read.\n', nreal);

for i = 1:numel(cases)
    x = cases(i).x; y = cases(i).y; mask = cases(i).mask;
    % myGaussFit and the no-toolbox branch
    [sigma,mu,s,y0] = myGaussFit(x(mask==1), y(mask==1));  mygauss = [sigma mu s y0];
    [xfit_legacy,yfit_legacy,waist_legacy] = gaussianfit_legacy(x,y,mask);
    % toolbox branch
    gauss1_ok = 1; coefs_gauss1 = [NaN NaN NaN NaN]; waist_gauss1 = NaN; xfit_gauss1 = []; yfit_gauss1 = [];
    try
        [xfit_gauss1,yfit_gauss1,waist_gauss1,coefs_gauss1] = gaussianfit_toolbox(x,y,mask);
        yfit_gauss1 = yfit_gauss1(:); xfit_gauss1 = xfit_gauss1(:);
    catch err
        gauss1_ok = 0; fprintf('  gauss1 failed on %s: %s\n', cases(i).name, err.message);
    end
    xfit_legacy = xfit_legacy(:); yfit_legacy = yfit_legacy(:);
    save(['profile_' regexprep(cases(i).name,'[^\w]','_') '.mat'],'x','y','mask','mygauss', ...
         'waist_legacy','xfit_legacy','yfit_legacy','gauss1_ok','coefs_gauss1','waist_gauss1', ...
         'xfit_gauss1','yfit_gauss1','-v7');
    fprintf('%-45s gauss1 waist=%-10.5g legacy waist=%-10.5g\n', cases(i).name, waist_gauss1, waist_legacy);
end

% =====================================================================
% COPIES of gaussianfit (branch hard-wired); coefs added as 4th output
% =====================================================================
function [xfit,yfit,waist,coefs]=gaussianfit_toolbox(x,y,linemask)
    xfit=[]; yfit=[]; waist=0;
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
    if waist~=0
        xfitstep=0.1*(x(length(x))-x(1))/(length(x)-1);
        xfit=(x(1):xfitstep:x(length(x)));
        yfit=f(xfit);
    end
end

function [xfit,yfit,waist]=gaussianfit_legacy(x,y,linemask)
    xfit=[]; yfit=[]; waist=0;
    [sigma, mu, amp, y0] = myGaussFit(x(linemask==1), y(linemask==1));
    f = @(x) y0 + (amp/(sqrt(2*pi)*sigma))*exp( -(x - mu).^2 ./ (2*sigma.^2));
    waist = 2*sigma;
    if waist~=0
        xfitstep=0.1*(x(length(x))-x(1))/(length(x)-1);
        xfit=(x(1):xfitstep:x(length(x)));
        yfit=f(xfit);
    end
end
