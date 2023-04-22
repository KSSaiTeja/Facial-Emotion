clc;
clear all;
close all;
warning off;
imds=imageDatastore('Mj','IncludeSubFolders',true,'LabelSource','foldernames', 'FileExtensions', '.jpg');
trainingFeatures=[];
trainingLabels=imds.Labels;       
for i = 1:numel(imds.Files)         % Read images using a for loop
    img = readimage(imds,i);
    trainingFeatures(i,:)=extractLBPFeatures(im2gray(img));
end
Classifier =fitcecoc(trainingFeatures,trainingLabels);
save Classifier Classifier