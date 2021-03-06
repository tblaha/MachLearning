clear
close all
         
%% process the txt --> matlab table (different from matrices!)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Quick explanation on how to use tables %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% you can get a column (as matlab vector) by typing data.cylinders
% you can get a column (as matlab table)  by typing data(:,2)
% you can get a list of column names by data.Properties.VariableNames

%%% import
data = readtable('Data.txt'); % read data
desc = {'gpm', 'cylinders', 'displacement', 'horsepower', 'weight', 'accelleration', 'year', 'origin', 'name'};
data.Properties.VariableNames = desc; % add descriptors as table column names

%%% invert mpg for better linear correlations
data.gpm = 1./data.gpm;
%%data.accelleration = 1./data.accelleration;
%%data.weight = 1./data.weight;

%%% fix horsepower column
% remove rows with unknown horsepower
data(strcmp(data.horsepower, '?'),:) = [];

% convert horsepower column to doubles instead of strings
len = height(data);
horse = zeros([len 1]);
for i = 1:len
    horse(i) = (str2double(cell2mat(data{i,'horsepower'})) );
end
data.horsepower = horse;



%%% add column of unique integer id's and move the names to second position
id = [1:len]';
data = [array2table(id), data(:,end), data(:,1:end-1)];




%% strip important data into observation matrix for linear algebra processing

X = data{:,3:end}; % all but name and id
[N,M] = size(X);
X(:,1:7) = (X(:,1:7) - mean(X(:,1:7))) ./ std(X(:,1:7));
% X = (X - mean(X)) ./ std(X);

% note that {} are used instead of () to not keep the 'table' data type

% matrix of only input arguments (excluding the fuel consumption)
Y = data{:,4:end};
[Ny,My] = size(Y);


%% clean up workspace

%save('X', 'X')

clear id horse len i