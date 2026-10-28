function [utils, samples, here, shims] = export_setup()
% Common set-up of the export_*.m scripts: locates Utils/ and SampleBeams/ RELATIVE TO THIS FOLDER
% (they are expected one level up: <project>/Python migration/export_*.m, <project>/Utils, <project>/SampleBeams),
% stops with an explicit error if one is missing (addpath alone only warns, and the exporters
% used to silently produce synthetic cases only), puts Utils on the path and moves to this folder,
% so that the .mat files are written next to the scripts.
% 4th output: path of ../shim (fake fittype etc., used only when the Curve Fitting Toolbox is absent);
% it is NOT added to the path here, and no error is raised if it does not exist.
here    = fileparts(mfilename('fullpath'));
utils   = fullfile(here, '..', 'Utils');
samples = fullfile(here, '..', 'SampleBeams');
shims   = fullfile(here, '..', 'shim');
if ~isfolder(utils)
    error('export_setup:noUtils', 'Utils folder not found: %s\n(expected one level above the export scripts)', utils);
end
if ~isfolder(samples)
    error('export_setup:noSamples', 'SampleBeams folder not found: %s\n(expected one level above the export scripts)', samples);
end
utils = char(java.io.File(utils).getCanonicalPath());      % clean path, without "..\"
samples = char(java.io.File(samples).getCanonicalPath());
shims = char(java.io.File(shims).getCanonicalPath());
addpath(utils);
cd(here);
fprintf('Utils:       %s\nSampleBeams: %s\nOutput:      %s\n', utils, samples, here);
end
