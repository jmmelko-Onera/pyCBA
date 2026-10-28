function [sigma, mu, s, y0] = myGaussFit( x, y, sigma0, mu0 )
% [sigma, mu, s, y0] = myGaussFit( x, y, sigma0, mu0 )
% Fits y0 + s * 1/(sqrt(2*pi)*sigma)*exp( -(x - mu)^2 / (2*sigma^2)) into (x,y) points,
% using an iterative LMS method (Gauss-Newton).
% The results are much better than minimazing logarithmic residuals
%
% INPUT:
% sigma0 - initial value of sigma (optional)
% mu0 - initial value of mean (optional)
%
% OUTPUT:
% sigma - optimal value of standard deviation
% mu - optimal value of mean
% s - amplitude (area under the curve once the baseline y0 is removed)
% y0 - baseline, estimated from the first and last points
%
% VERSION: 23.02.2012
% 18/10/2026 - Corrections:
%   - the data are always divided by s (they were only when s<0.5 or s>1.5), so that the
%     returned s is the true amplitude of the fit, for any s
%   - sigma0 and mu0 are handled independently (when both were given, only mu0 was used)
%   - Gauss-Newton with step halving: a step is only taken if sigma stays positive and the
%     residual does not increase. No more divergence / "Instability detected"
%   - the MATLAB warnings turned off during the fit are restored on exit (they stayed off
%     after an early return)
%   - returns sigma = mu = 0 if the area is null

sigma = 0; mu = 0; s = 0; y0 = 0;
Nmax = 50; % Maximum number of iterations
if length(x) ~= length(y), return; end
n = length(x);
if n==0, return; end
x = reshape(x, n, 1);
y = reshape(y, n, 1);
%sort according to x
X = sortrows([x,y]);
x = X(:,1);
y = X(:,2);

%substracting baseline
if n>13
    y0 = (mean(y(1:5)) + mean(y(end-4:end)))/2;
else
    y0 = (y(1) + y(end))/2;
end
y = y - y0;

%normalization
dx = diff(x);
dy = 0.5*(y(1:length(y)-1) + y(2:length(y)));
s = sum(dx .* dy);
if s == 0 || ~isfinite(s), return; end
y = y ./ s;

% estimate mean mu from the location of the maximum, and sigma from its height
[ymax,index] = max(y);
mu = x(index);
sigma = 1/(sqrt(2*pi)*ymax);
if nargin>=3 && ~isempty(sigma0), sigma = sigma0; end
if nargin>=4 && ~isempty(mu0), mu = mu0; end

ws = [warning('off','MATLAB:illConditionedMatrix'); warning('off','MATLAB:singularMatrix')];
restoreWarnings = onCleanup(@() warning(ws));
gausspdf = @(sg,m) 1/(sqrt(2*pi)*sg) * exp( -((x-m).^2) / (2*sg^2));

for i=1:Nmax
    e = exp(-((x-mu).^2) / (2*sigma^2));
    dfdsigma = -1/(sqrt(2*pi)*sigma^2)*e + 1/(sqrt(2*pi)*sigma).*e.*((x-mu).^2/sigma^3);
    dfdmu = 1/(sqrt(2*pi)*sigma).*e.*(x-mu)/(sigma^2);
    F = [dfdsigma dfdmu];
    res = y - gausspdf(sigma,mu);
    r0 = sum(res.^2);
    step = (F'*F)\(F'*res);
    if any(~isfinite(step)), break; end
    t = 1; accepted = false;
    for k=1:10 % step halving
        sn = sigma + t*step(1);
        mn = mu + t*step(2);
        if sn > 0 && sum((y - gausspdf(sn,mn)).^2) <= r0
            accepted = true;
            break;
        end
        t = t/2;
    end
    if ~accepted, break; end
    converged = abs(sn-sigma) <= 1e-12*sigma && abs(mn-mu) <= 1e-12*sigma;
    sigma = sn;
    mu = mn;
    if converged, break; end
end
end
