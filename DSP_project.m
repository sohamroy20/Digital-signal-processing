clc
clear all
close all
%sample data and sample rate
[audioIn, fs] = audioread('Counting-16-44p1-mono-15secs.wav');
audioStart = 110e3;
audioStop = 135e3;
audioIn = audioIn(audioStart:audioStop);
timeVector = linspace((audioStart/fs),(audioStop/fs),numel(audioIn));
sound(audioIn,fs)
figure
plot(timeVector,audioIn)
axis([(audioStart/fs) (audioStop/fs) -1 1])
ylabel('Amplitude')
xlabel('Time (s)')
title('Utterance')


windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);

%Using the pitch function we see how pitch changes over time.
f0=pitch(audioIn,fs,'WindowLength', windowLength,'OverlapLength',overlapLength,'Range',[50,250]);
figure
subplot(2,1,1)
plot(timeVector,audioIn)
axis([(110e3/fs) (135e3/fs) -1 1])
ylabel('Amplitude')
xlabel('Time (s)')
title('Utterance')
subplot(2,1,2)
timeVectorPitch = linspace((audioStart/fs),(audioStop/fs),numel(f0));
plot(timeVectorPitch,f0,'*')
axis([(110e3/fs) (135e3/fs) min(f0) max(f0)])
ylabel('Pitch (Hz)')
xlabel('Time (s)')
title('Pitch Contour')

%to distinguish between silence and speech is to analyse the power. If the power in a frame is above a given threshold, you declare the frame as speech
pwrThreshold = -20;
[segments,~] = buffer(audioIn,windowLength, overlapLength,'nodelay');
pwr = pow2db(var(segments));
isSpeech = (pwr > pwrThreshold);

%analysing zerocrossingrate to distinguish voiced and unvoiced
zcrThreshold = 300;
zeroLoc = (audioIn==0);
crossedZero = logical([0;diff(sign(audioIn))]);
crossedZero(zeroLoc) = false;
[crossedZeroBuffered,~] = buffer(crossedZero,windowLength,overlapLength,'nodelay');
zcr = (sum(crossedZeroBuffered,1)*fs)/(2*windowLength);
isVoiced = (zcr < zcrThreshold);

%combining isSpeech and isVoice

voicedSpeech = isSpeech & isVoiced;
f0(~voicedSpeech) = NaN;
figure
subplot(2,1,1)
plot(timeVector,audioIn)
axis([(110e3/fs) (135e3/fs) -1 1])
axis tight
ylabel('Amplitude')
xlabel('Time (s)')
title('Utterance')

subplot(2,1,2)
plot(timeVectorPitch,f0,'*')
axis([(110e3/fs) (135e3/fs) min(f0) max(f0)])
ylabel('Pitch (Hz)')
xlabel('Time (s)')
title('Pitch Contour')




%MFCC
dataDir = HelperAN4Download;
ads = audioDatastore(dataDir,'IncludeSubfolders',true, ...
    'FileExtensions','.flac', ...
    'LabelSource','foldernames')
[adsTrain, adsTest] = splitEachLabel(ads,0.8);
adsTrain
trainDatastoreCount = countEachLabel(adsTrain)
adsTest
testDatastoreCount = countEachLabel(adsTest)
[sampleTrain, dsInfo] = read(adsTrain);
sound(sampleTrain,dsInfo.SampleRate)
reset(adsTrain)

%Feature Extraction
fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);

features = [];
labels = [];
while hasdata(adsTrain)
    [audioIn,dsInfo] = read(adsTrain);
    
    melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
    f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
    feat = [melC,f0];
    
    voicedSpeech = isVoicedSpeech(audioIn,fs,windowLength,overlapLength);
    
    feat(~voicedSpeech,:) = [];
    label = repelem(dsInfo.Label,size(feat,1));
    
    features = [features;feat];
    labels = [labels,label];
end


% Normalizing the features
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;


% Training the classifier
trainedClassifier = fitcknn( ...
    features, ...
    labels, ...
    'Distance','euclidean', ...
    'NumNeighbors',5, ...
    'DistanceWeight','squaredinverse', ...
    'Standardize',false, ...
    'ClassNames',unique(labels));

% Performing cross-validation
k = 5;
group = labels;
c = cvpartition(group,'KFold',k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',c);

% Computing validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel,'LossFun','ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

% Confusion chart
validationPredictions = kfoldPredict(partitionedModel);
figure
cm = confusionchart(labels,validationPredictions,'title','Validation Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';


% Testing the classifier
features = [];
labels = [];
numVectorsPerFile = [];
while hasdata(adsTest)
    [audioIn,dsInfo] = read(adsTest);
    
    melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
    f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
    feat = [melC,f0];
    
    voicedSpeech = isVoicedSpeech(audioIn,fs,windowLength,overlapLength);
    
    feat(~voicedSpeech,:) = [];
    numVec = size(feat,1);
    
    label = repelem(dsInfo.Label,numVec);
    
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features = [features;feat];
    labels = [labels,label];
end
features = (features-M)./S;

% Predicting speaker
prediction = predict(trainedClassifier,features);
prediction = categorical(string(prediction));


% Visualize confusion chart
figure('Units','normalized','Position',[0.4 0.4 0.4 0.4])
cm = confusionchart(labels,prediction,'title','Test Accuracy (Per Frame)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% Determining mode of predictions and plotting confusion chart
r2 = prediction(1:numel(adsTest.Files));
idx = 1;
for ii = 1:numel(adsTest.Files)
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end

figure('Units','normalized','Position',[0.4 0.4 0.4 0.4])
cm = confusionchart(adsTest.Labels,r2,'title','Test Accuracy (Per File)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% Supporting functions
function voicedSpeech = isVoicedSpeech(x,fs,windowLength,overlapLength)

pwrThreshold = -40;
[segments,~] = buffer(x,windowLength,overlapLength,'nodelay');
pwr = pow2db(var(segments));
isSpeech = (pwr > pwrThreshold);

zcrThreshold = 1000;
zeroLoc = (x==0);
crossedZero = logical([0;diff(sign(x))]);
crossedZero(zeroLoc) = false;
[crossedZeroBuffered,~] = buffer(crossedZero,windowLength,overlapLength,'nodelay');
zcr = (sum(crossedZeroBuffered,1)*fs)/(2*windowLength);
isVoiced = (zcr < zcrThreshold);

voicedSpeech = isSpeech & isVoiced;

end



% Helper AN4Download
function dataDir = HelperAN4Download
%HelperAN4Download Download and extract the AN4 dataset. 
% The dataset is downloaded from:
%  http://www.speech.cs.cmu.edu/databases/an4
%
% The speech files corresponding to 5 male and 5 female speakers are
% retained. The rest of the speech files are deleted. The output of the
% function is the path to the 'an4' parent directory in the dataset.
%
% This function HelperAN4Download is only in support of
% SpeakerIdentificationExample. It may change in a future release.

%   Copyright 2017 The MathWorks, Inc.

url = 'http://www.speech.cs.cmu.edu/databases/an4/an4_raw.littleendian.tar.gz';
d = tempdir;
fs = 16e3;

% Download and extract if it hasn't already been done.
unpackedData = fullfile(d, 'an4');
d1 = fullfile(unpackedData, 'wav','flacData');
if ~exist(unpackedData, 'dir')
    fprintf('Downloading AN4 dataset... ');
    untar(url, d);
    fprintf('done.\n');
    mkdir(d1);
end

% Reduce dataset to 5 female and 5 male speakers
validDirs = {'fejs', 'fmjd', 'fsrb', 'ftmj', 'fwxs', ...
             'mcen', 'mrcb', 'msjm', 'msjr', 'msmn', '.', '..'};
d3 = fullfile(unpackedData, 'wav', 'an4_clstk');
listing = dir(d3);
l = {listing(:).name};

if length(l) > length(validDirs)
    fprintf('Reducing dataset to 5 females and 5 males... ');
    for idx = 3:length(l)
        if ~ismember(l{idx}, validDirs)
            rmdir(fullfile(d3,l{idx}),'s');
        else
           mkdir(fullfile(d1,l{idx}));
           list = dir(fullfile(d3,l{idx}));
           for i = 3:length(list)
               filename = list(i).name;
               fname = fullfile(d3,l{idx},filename);
               
               fullfilename = fullfile(d1,l{idx},filename);
               
               % % Read binary data (stored as int16)
               
               fid = fopen(fname,'r');
               xint = int16(fread(fid,[1,inf],'int16')).';
               fclose(fid);
               
               % Scale int16 to double
               x = double(xint)*2^-15;
               % % Convert to flac and write it
               newname = strrep(fullfilename,'.raw','.flac');
               audiowrite(newname,x,fs);
           end
        end
    end
    rmdir(fullfile(unpackedData, 'wav', 'an4test_clstk'),'s');
    fprintf('done.\n');
end

dataDir = d1;

end

