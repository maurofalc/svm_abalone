% Reconhecimento de Padrões
% SVM - Abalone Dataset

clear;
clc;
close all;

tbl = readtable('abalone.data', 'Filetype', 'text', ...
    'ReadVariableNames', false);

samples = tbl(:, 2:9);
class1 = tbl(:, 1);
class2 = tbl(:, 1);
class3 = tbl(:, 1);

% Quantidade de amostras.
N = height(samples);

for i = 1:N
    % Faz com que todos da classe M e F virem classe MF
    if tbl.Var1{i} == 'M' || tbl.Var1{i} == 'F'
        class1.Var1{i} = 'MF';
    end
    % Faz com que todos da classe M e I virem classe MI
    if tbl.Var1{i} == 'M' || tbl.Var1{i} == 'I'
        class2.Var1{i} = 'MI';
    end
    % Faz com que todos da classe I e F virem classe IF
    if tbl.Var1{i} == 'I' || tbl.Var1{i} == 'F'
        class3.Var1{i} = 'IF';
    end
end

% Preparação dos dados para treino.
data1 = [class1 samples];
data2 = [class2 samples];
data3 = [class3 samples];

% Treinamento de cada classificador.
mdl1 = fitcsvm(data1(2:N, :), 'Var1', 'KernelFunction', ...
    'gaussian', 'KernelScale', 2.2, 'Standardize', true);
mdl2 = fitcsvm(data2(2:N, :), 'Var1', 'KernelFunction', ...
    'gaussian', 'KernelScale', 2.2, 'Standardize', true);
mdl3 = fitcsvm(data3(2:N, :), 'Var1', 'KernelFunction', ...
    'gaussian', 'KernelScale', 2.2, 'Standardize', true);


% Predição da primeira amostra de cada classificador.
[label1, score1, cost1] = predict(mdl1, samples(1, :));
[label2, score2, cost2] = predict(mdl2, samples(1, :));
[label3, score3, cost3] = predict(mdl3, samples(1, :));

% fprintf("1 Classe predita: %s\tScore: %f\n", string(label1), score1(:, 1));
% fprintf("2 Classe predita: %s\tScore: %f\n", string(label2), score2(:, 1));
% fprintf("3 Classe predita: %s\tScore: %f\n", string(label3), score3(:, 1));

% CVMdl = crossval(mdl);
% openExample('stats/TrainSupportVectorMachineRegressionModelExample');