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

figure();
subplot(2,1,1);
imshow(AxSeg);
subplot(2,1,2);
imshow(AxSeg_rb);

[im_out] = myelinInitialSegmention(im_in, AxSeg_rb, backBW,0,1,PixelSize);
im_out = myelinCleanConflict(im_out, im_in,0.5);

axonlist = as_myelinseg2axonlist(im_out,PixelSize);

save axonlist.mat axonlist

imwrite(sum(im_out,3),'MyelinSeg.jpg')

end


    
% [tmp,border_removed_mask]=RemoveBorder(handles.data.Step3_seg,get(handles.PixelSize,'Value'));
% backBW=handles.data.Step3_seg & ~tmp;
% 
% end
% 
% [handles.data.seg] = myelinInitialSegmention(handles.data.Step1, tmp, backBW,0,1,get(handles.PixelSize,'Value'));
% handles.data.seg = myelinCleanConflict(handles.data.seg,1,0.5);
% 
% handles.data.Step3_seg = as_myelinseg_to_axonseg(handles.data.seg);
% 
% axes(handles.plotseg);
% sc(sc(handles.data.Step1)+sc(sum(handles.data.seg,3),'copper')+sc(border_removed_mask,[0.5 0.4 0.4], border_removed_mask));
% 
% handles.data.labelseg=zeros(size(handles.data.seg,1), size(handles.data.seg,2));
% for i=1:size(handles.data.seg,3)
%     handles.data.labelseg(logical(handles.data.seg(:,:,i)))=i;
% end
