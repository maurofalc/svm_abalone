% Reconhecimento de Padrões
% SVM - Abalone Dataset

clear;
clc;
close all;

tbl = readtable('abalone.data', 'Filetype', 'text', ...
    'ReadVariableNames', false);

% Faz com que todos da classe M e F virem classe MF
for i = 1:height(tbl)
    if tbl.Var1{i} == 'M' || tbl.Var1{i} == 'F'
        tbl.Var1{i} = 'MF';
    end
end

mdl = fitcsvm(tbl(2:height(tbl), :), 'Var1', 'KernelFunction', ...
    'gaussian', 'KernelScale', 2.2, 'Standardize', true);

% Predição da primeira amostra
[label, score] = predict(mdl, tbl(1, 2:9));
fprintf("Classe predita: %s\n", string(label));

% CVMdl = crossval(mdl);
% openExample('stats/TrainSupportVectorMachineRegressionModelExample');