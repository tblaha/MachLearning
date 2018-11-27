%% Association Mining
clc;clear;
if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('XoneoutofK.mat');
    data=load('X.mat');
    X=data.X;
else
    %run(importdata_Report2.m); %For K out of N
    run(importdata_Report1.m); %For K
end
attributeNames={'gpm','cylinders', 'displacement', 'horsepower', 'weight', 'accelleration', 'year', 'origin'};
[Xbin,attributeNamesBin] = binarize(X(:,:), 3*ones(1,size(X,2)), attributeNames);

%Classes = 3
%Cylinders = 5
%Continous = 

%%
minSup = 0.3; % minimum support
minConf = .81; % minmum confidence
nRules = 100; % Max rules
sortFlag = 1; % sorting of found rules (see doc)
[rules, ~] = findRules(Xbin, minSup, minConf, nRules, sortFlag);
disp('Rules found:')
print_apriori_rules(rules,attributeNamesBin)