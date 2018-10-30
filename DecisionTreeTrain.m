clc, clear;
close all;
%Importing X depending on MATLAB version
if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('XoneoutofK.mat');
    data=load('X.mat');
    X=data.X;
else
    run(importdata_Report2.m); %For K out of N
    %run(importdata_Report1.m); %For K
end

%Setup for the fitctree
classNames= {'USA', 'Europe', 'Asia'}';
attributeNames = {'gpm', 'cylinders', 'displacement', 'horsepower', 'weight', 'accelleration'};
y=zeros(size(X,1),1);

if(size(X,2)>8)     %If X one out of K is runned, else it's normal X
    for i = 1:size(X,1)
        if(X(i,8)==1)
            y(i)=1;
        elseif(X(i,9)==1)
            y(i)=2;
        elseif(X(i,10)==1)
            y(i)=3;
        end
    end
    for i=1:4   %Removing origins and year
        X(:,7)=[];
    end
else
   y=X(:,8);    %Saving the origins to be used for classNames
   for i=1:2   %Removing origins and year
        X(:,7)=[];
    end
end

%Boxplot to find outliers
%boxplot(X, attributeNames, 'LabelOrientation', 'inline');
%boxplot(zscore(X), attributeNames, 'LabelOrientation', 'inline');

%Removing outliers from data set
idxOutlier = find(X(:,4)>200 | X(:,6)>21.9 | X(:,6)<9 | X(:,1)>0.09);
X(idxOutlier,:) = [];
y(idxOutlier) = [];

% View the tree
T2 = fitctree(X, classNames(y), ...
   'splitcriterion', 'gdi', ...
   'categorical', [], ...
   'PredictorNames', attributeNames, ...
   'prune', 'off', ...
   'minparent', 50);
view(T2, 'Mode','graph');



