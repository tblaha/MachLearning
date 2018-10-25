clc, clear;
close all;
%Importing X depending on MATLAB version
if(version()==('9.5.0.944444 (R2018b)'))
    data=load('XoneoutofK.mat');
    % X=load('X.mat');
    X=data.X;
else
    run(importdata_Report2.m); %For K out of N
    %run(importdata_Report1.m); %For K
end

classNames= {'USA', 'Europe', 'Asia'};
attributeNames = {'gpm', 'cylinders', 'displacement', 'horsepower', 'weight', 'accelleration', 'year', 'USA', 'Europe', 'Asia'};
%Boxplot to find outliers
%fig('Car: Boxplot');%Checking outliers for each attribute, can discard the origin
boxplot(zscore(X), attributeNames, 'LabelOrientation', 'inline');
% View the tree
view(T2, 'Mode','graph');



