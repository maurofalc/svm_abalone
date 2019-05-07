%% Reconhecimento de Padrões
% SVM - Abalone Dataset

clear;
clc;
close all;

%% Leitura da base de dados.
tbl = readtable('abalone.data', 'Filetype', 'text', ...
    'ReadVariableNames', true);

%% Separação dos dados.
samples = tbl(:, 2:9);
classes = tbl(:, 1);
data1 = classes;
data2 = classes;
data3 = classes;
classes = cell2mat(table2array(classes));

%% Parâmetros
N = height(samples); % Quantidade de amostras.
K = 10; % Tamanho do K-fold.
kernel = 'rbf';
scale = 2.2;

%% Preparação das classes e dados
for i = 1:N
    % Faz com que todos da classe M e F virem classe MF
    if tbl.Sex{i} == 'M' || tbl.Sex{i} == 'F'
        data1.Sex{i} = 'MF';
    end
    % Faz com que todos da classe M e I virem classe MI
    if tbl.Sex{i} == 'M' || tbl.Sex{i} == 'I'
        data2.Sex{i} = 'MI';
    end
    % Faz com que todos da classe I e F virem classe IF
    if tbl.Sex{i} == 'I' || tbl.Sex{i} == 'F'
        data3.Sex{i} = 'IF';
    end
end

% Preparação dos dados para treino.
data1 = [data1 samples];
data2 = [data2 samples];
data3 = [data3 samples];

%% K-Fold
% Indices do K-Fold.
kfold = crossvalind('Kfold', N, K);
acertos = zeros(K, 1);

predictedClass = [];
realClass = [];

for k = 1:K
    %% Separa os indices de treino e teste.
    teste = (kfold == k);
    treino = ~teste;
    
    %% Treinamento de cada classificador.
    mdl1 = fitcsvm(data1(treino, :), 'Sex', 'KernelFunction', ...
        kernel, 'KernelScale', scale, 'Standardize', true);
    mdl2 = fitcsvm(data2(treino, :), 'Sex', 'KernelFunction', ...
        kernel, 'KernelScale', scale, 'Standardize', true);
    mdl3 = fitcsvm(data3(treino, :), 'Sex', 'KernelFunction', ...
        kernel, 'KernelScale', scale, 'Standardize', true);

    %% Predição das amostras de teste para cada classificador.
    [~, score1] = predict(mdl1, samples(teste, :));
    [~, score2] = predict(mdl2, samples(teste, :));
    [~, score3] = predict(mdl3, samples(teste, :));

    % Organização dos scores.
    scores = array2table([score1 score2 score3], 'VariableNames', ...
        [mdl1.ClassNames; mdl2.ClassNames; mdl3.ClassNames]);

    % Classe mais provável.
    ml = mlclass(scores);
    
    %% Cálculo da porcentagem de acerto da rodada.
    acertos(k) = sum(ml == classes(teste))/sum(teste)*100;
    fprintf("Acerto K=%2d: %2.2f%%\n", k, acertos(k));
    
    predictedClass = [predictedClass; ml];
    realClass = [realClass; classes(teste)];
end

fprintf("Acerto Médio: %2.2f%%\n", mean(acertos));

%% Gráficos
% Confusion
target = zeros(3, length(predictedClass));
out = target;

for i = 1:length(predictedClass)
    if predictedClass(i) == 'M'
        out(:,i) = [1 0 0]';
    end
    if predictedClass(i) == 'F'
        out(:,i) = [0 1 0]';
    end
    if predictedClass(i) == 'I'
        out(:,i) = [0 0 1]';
    end

    if realClass(i) == 'M'
        target(:,i) = [1 0 0]';
    end
    if realClass(i) == 'F'
        target(:,i) = [0 1 0]';
    end
    if realClass(i) == 'I'
        target(:,i) = [0 0 1]';
    end
end

plotconfusion(target, out);
set(gca,'xticklabel',{'M' 'F' 'I' ''});
set(gca,'yticklabel',{'M' 'F' 'I' ''});

% Acertos por rodada
figure;
hold on;
grid on;
ylabel('Porcentagem de Acerto');
xlabel('Valor de K');
title('Porcentagem de Acertos K-Fold');
set(legend('show'), 'Location', 'best');
% ylim([0 100]);
plot(acertos, 'DisplayName', 'Acertos', 'MarkerSize', 20, 'Marker', ...
    '.', 'LineWidth', 2);
plot(mean(acertos)*ones(1,K), 'DisplayName', 'Acerto Médio', ...
    'LineStyle', '--', 'LineWidth', 2);