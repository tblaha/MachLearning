%% Association Mining
clc;clear;

data=load('XoneoutofK.mat');
%data=load('X.mat');
X=data.X;
attributeNames1={'gpm'};
attributeNames2={'displacement', 'horsepower', 'weight', 'accelleration', 'year',};
attributeNames3={'3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders'};
attributeNames4={'USA','Europe','Asia'};
[Xbin1,attributeNamesBin1] = binarize(X(:,1), 3*ones(1,size(X,2)), attributeNames1);
[Xbin2,attributeNamesBin2] = binarize(X(:,3:7), 3*ones(1,size(X,2)), attributeNames2);

%Classes = 3
%Cylinders = 5
%Continous = 

%% one-out-of-K cylinders
datann=load('Xnn.mat'); 
Xnn=datann.Xnn; %not normalized data
X_cylinders=zeros(size(Xnn,1),5);  % 5 different posssible values

for i=1:size(Xnn,1)
    switch Xnn(i,2) 
        case 3
            j=1;
        case 4
            j=2;
        case 5
            j=3;
        case 6
            j=4;
        case 8
            j=5;
    end    
    X_cylinders(i,j)=1;
end
%% total binarized matrix X

Xbin=[Xbin1,X_cylinders,Xbin2,X(:,8:10)];
attributeNamesBin=[attributeNamesBin1,attributeNames3,attributeNamesBin2,attributeNames4];
%%
minSup = 0.32; % minimum support
minConf = 1; % minmum confidence
nRules = 100; % Max rules
sortFlag = 1; % sorting of found rules (see doc)
[rules, ~] = findRules(Xbin, minSup, minConf, nRules, sortFlag);
disp('Rules found:')
print_apriori_rules(rules,attributeNamesBin)