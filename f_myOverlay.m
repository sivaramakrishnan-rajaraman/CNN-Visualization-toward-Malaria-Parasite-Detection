%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dr. Stefan Jaeger (c) 2014
%
% National Library of Medicine
% National Institutes of Health
% 8600 Rockville Pike
% Bethesda, MD 20894, USA
% Email: stefan.jaeger@nih.gov
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Overlay of probability map onto image
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% function outputImage = f_myOverlay(I,probabilityMap,colorThreshold,mergeRatio)
%
%Input:
%- I, input image
%- probabilityMap, probability map to be overlayed
%- colorThreshold, threshold to cut off lower part of jet color map
%- mergeRatio, determines the visibility of I (1: only I is visible,
%  0: only probability map is visible)
%
%Output:
%- outputImage, output overlay image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function outputImage = f_myOverlay(I,probabilityMap,colorThreshold,mergeRatio)

colorThreshold = round(colorThreshold);

if (colorThreshold > 256)
    colorThreshold = 256;
end

SM = uint8(probabilityMap*255);
CM = colormap(jet(256));
if (colorThreshold >= 1)
    CM(1:colorThreshold,:) = 0;
end
RGB = ind2rgb(SM,CM);
outputImage = my_immerge(RGB,I,mergeRatio);

return