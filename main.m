% Reconhecimento de Padrões
% SVM - Abalone Dataset

clear;
clc;
close all;

tbl = readtable('abalone.data', 'Filetype', 'text', ...
    'ReadVariableNames', false);

samples = tbl(:, 2:9);
data1 = tbl(:, 1);
data2 = tbl(:, 1);
data3 = tbl(:, 1);

% Quantidade de amostras.
N = height(samples);

for i = 1:N
    % Faz com que todos da classe M e F virem classe MF
    if tbl.Var1{i} == 'M' || tbl.Var1{i} == 'F'
        data1.Var1{i} = 'MF';
    end
    % Faz com que todos da classe M e I virem classe MI
    if tbl.Var1{i} == 'M' || tbl.Var1{i} == 'I'
        data2.Var1{i} = 'MI';
    end
    % Faz com que todos da classe I e F virem classe IF
    if tbl.Var1{i} == 'I' || tbl.Var1{i} == 'F'
        data3.Var1{i} = 'IF';
    end
end

% Preparação dos dados para treino.
data1 = [data1 samples];
data2 = [data2 samples];
data3 = [data3 samples];

% Treinamento de cada classificador.
mdl1 = fitcsvm(data1(6:N, :), 'Var1', 'KernelFunction', ...
    'gaussian', 'KernelScale', 2.2, 'Standardize', true);
mdl2 = fitcsvm(data2(6:N, :), 'Var1', 'KernelFunction', ...
    'gaussian', 'KernelScale', 2.2, 'Standardize', true);
mdl3 = fitcsvm(data3(6:N, :), 'Var1', 'KernelFunction', ...
    'gaussian', 'KernelScale', 2.2, 'Standardize', true);


% Predição da primeira amostra de cada classificador.
[~, score1] = predict(mdl1, samples(1:5, :));
[~, score2] = predict(mdl2, samples(1:5, :));
[~, score3] = predict(mdl3, samples(1:5, :));

scores = array2table([score1 score2 score3], 'VariableNames', ...
    [mdl1.ClassNames; mdl2.ClassNames; mdl3.ClassNames]);

ml = mlclass(scores);

% CVMdl = crossval(mdl);
% openExample('stats/TrainSupportVectorMachineRegressionModelExample');