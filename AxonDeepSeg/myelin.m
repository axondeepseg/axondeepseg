function [] = myelin(path, PixelSize)

cd (path)

path_Mask = 'AxonMask';
path_img = 'image.jpg';

load (path_Mask)

Iinitial = imread(path_img);

[rows columns numberOfColorChannels] = size(Iinitial);
if numberOfColorChannels > 1
  im_in = rgb2gray(Iinitial);
else
  im_in = Iinitial;
end

AxSeg = double(prediction);

% Myelin Segmentation

[AxSeg_rb,~]=RemoveBorder(AxSeg,PixelSize);
backBW=AxSeg & ~AxSeg_rb; % backBW = axons that have been removed by RemoveBorder


[im_out] = myelinInitialSegmention(im_in, AxSeg_rb, backBW,0,1,PixelSize);

im_out = myelinCleanConflict(im_out, im_in,0.5);
axonlist = as_myelinseg2axonlist(im_out,PixelSize)

save axonlist.mat axonlist

imwrite(sum(im_out,3),'MyelinSeg.jpg')

end

