% Exports pal2colormap results for several forced sizes (see tests/test_palette.py).
% Run it with pal2colormap_fixed.m (renamed pal2colormap.m, or replacing your copy) on the path, from a folder containing Spiricon_128bit_RIFF.pal
% (or edit palfile). Writes palette_all.mat; copy it to pyCBA/tests/matlab_reference/ .
warning('off','all');
palfile = 'Spiricon_128bit_RIFF.pal';
sizes = [128 64 100 256 2 3]; %#ok<NASGU>
out = struct();
for s = [128 64 100 256 2 3]
    out.(sprintf('cm_%d',s)) = pal2colormap(palfile,s);
end
out.sizes = [128 64 100 256 2 3];
save('palette_all.mat','-struct','out');
disp('palette_all.mat written');
