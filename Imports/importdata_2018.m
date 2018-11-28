if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('XoneoutofK.mat');
    data=load('X.mat');
    X=data.X;
    datann=load('Xnn.mat'); % not normalizes values
    Xnn=datann.Xnn;
else
    %run(importdata_Report2.m); %For K out of N
    run(importdata_Report1.m); %For K
end