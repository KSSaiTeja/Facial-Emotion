clearvars;
warning('off','all');

load Classifier;
disp('loaded');
camera = webcam;

faceDetector = vision.CascadeObjectDetector;

while true
    image = snapshot(camera);
    bboxes = step(faceDetector, image);
    if isempty(bboxes)
        imshow(image);
        title('Face Not Detected');
    else
        if size(bboxes, 1) > 1
            bboxes = bboxes(1, :);
        end
        es = imcrop(image, bboxes);
        es = imresize(es, [128 128]);
        es = rgb2gray(es);
        Features = extractLBPFeatures(es);
        PredictedClass = predict(Classifier, Features);
        PredictedClass = char(PredictedClass);
        imshow(image);
        title(PredictedClass);
    end
    drawnow;
    pause(0.1);
end
