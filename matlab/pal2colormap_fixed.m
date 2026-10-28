% Imports Microsoft Binary (RIFF) palette to Matlab RGB(0-1) colormap: [colormap{,RGBPalette}]=pal2colormap(palettepath{,forcedsize})
% Fixes: (1) the unused "Displays palette" block built an image of numcolors rows and filled it with the
% forced-size palette: any forcedsize ~= file size made the function fail; (2) on error the second output was
% stored in varargout{2} instead of varargout{1}.
function [colormap,varargout]=pal2colormap(palettepath,varargin)
if ~ischar(palettepath)
    colormap=[];
    if nargout>=2 varargout{1}=[]; end
    warning('Palette path is not a string array.');
else
    fid=fopen(palettepath);
    if fid<0
        colormap=[];
        if nargout>=2 varargout{1}=[]; end
        warning(['Cannot find/open ' palettepath '.']);
    else
        % Skip PAL Headers
        fseek(fid, 4, 'bof'); % Skips 'RIFF'
        fseek(fid, 4, 'cof'); % Skips file length
        fseek(fid, 4, 'cof'); % Skips 'PAL '
        fseek(fid, 4, 'cof'); % Skips 'file'
        fseek(fid, 4, 'cof'); % Skips data block size
        fseek(fid, 2, 'cof'); % Skips version
        numcolors=fread(fid,1,'ushort=>double');

        colormap=zeros(numcolors,3);
        for i=1:numcolors
            colormap(i,:)=fread(fid,3)/255;
            fread(fid,1);
        end

        fclose(fid);

        disp(sprintf('Imported RIFF palette with %d colors.',numcolors));
        [f,e] = log2(numcolors);
        if f~=0.5
            warning('Number of colors in palette is not 2^n. Are you sure it is a valid color palette?');
        end

        % Forced size conversion
        if nargin>=2
            forceddepth=varargin{1};
            if forceddepth~=numcolors
                x=(0:numcolors-1)/(numcolors-1);
                xi=(0:forceddepth-1)/(forceddepth-1);
                colormapHSV=rgb2hsv(colormap);
                forcedcolormapHSV=interp1(x,colormapHSV,xi,'linear');
                forcedcolormap=hsv2rgb(forcedcolormapHSV);
                colormap=forcedcolormap;
                disp(sprintf('Forced palette to %d colors.',forceddepth));
                if forceddepth>numcolors
                    warning('Forced color depth is longer than original palette depth. Interpolation may fail.');
                end
            end
        end

        if nargout>=2
            varargout{1}=255*colormap;   % RGB palette, 0-255
        end
    end
end
end
