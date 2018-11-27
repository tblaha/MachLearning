clc; clear all;
% Clustering
if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('../XoneoutofK.mat');
    data=load('../X.mat');
    X=data.X;
else
    %importdata_Report2; %For K out of N
    importdata_Report1; %For K
end;

cl