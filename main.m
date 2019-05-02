% Reconhecimento de Padrões
% SVM - Abalone Dataset

tbl = readtable('abalone.data','Filetype','text','ReadVariableNames',false);

mdl = fitrsvm(tbl,'Var9','KernelFunction','gaussian','KernelScale',2.2,'Standardize',true);

CVMdl = crossval(mdl);

%openExample('stats/TrainSupportVectorMachineRegressionModelExample');