% Run it from its own folder: Utils/ and SampleBeams/ are looked for in ../Utils and ../SampleBeams (export_setup.m).
% Produces ref_<filename with every non-word character replaced by "_">.mat (same rule as core_ and profile_) next to this script; copy them into
% pyCBA/tests/matlab_reference/ .
[~, sampleDir] = export_setup();
delete('ref_*.mat');   % stale files of a previous run must not survive
files = dir(fullfile(sampleDir,'*'));
files = files(~[files.isdir]);
if isempty(files), error('No file found in %s', sampleDir); end
for i = 1:numel(files)
    p = fullfile(sampleDir, files(i).name);
    try
        [numheaders,numlines,numcols,linedel,lineskip,coldel,lastcoldel,decsep,fixedcolsep] = analyzeASCII(p);
        r = readAsciiFromFile(p);
        save(['ref_' regexprep(files(i).name,'[^\w]','_') '.mat'], 'numheaders','numlines','numcols','linedel', ...
             'lineskip','coldel','lastcoldel','decsep','fixedcolsep','r');
        fprintf('%-45s size(r)=%s numlines=%d numcols=%d\n', files(i).name, mat2str(size(r)), numlines, numcols);
    catch err
        fprintf('%-45s FAILED: %s\n', files(i).name, err.message);
    end
end

% --- Two suspicions to confirm (see README) ---
r = readAsciiFromFile(fullfile(sampleDir,'sample_beam_SP928.csv'));
fprintf('\nSP928: size = %s (file has 5 rows x 1928 cols)\n', mat2str(size(r)));
fprintf('SP928: r(1,1) = %g, r(1,end) = %g (true last value of row 1 is -10.015625)\n', r(1,1), r(1,end));
