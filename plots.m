clear;
clc;
close all;

abalone = importdata("abalone.data");
data = zscore(abalone.data')';
class = abalone.textdata(2:4178,1);

classm = class == "M";
classf = class == "F";
classi = class == "I";

figure;
hold on;
grid on;

zlabel('Height');
ylabel('Diameter');
xlabel('Length');
title('Abalone');

set(legend('show'), 'Location', 'best');

plot3(data(classm, 1), data(classm, 2), data(classm, 3), '.', ...
    'DisplayName', 'Class M', 'MarkerSize', 10);
plot3(data(classf, 1), data(classf, 2), data(classf, 3), '.', ...
    'DisplayName', 'Class F', 'MarkerSize', 10);
plot3(data(classi, 1), data(classi, 2), data(classi, 3), '.', ...
    'DisplayName', 'Class I', 'MarkerSize', 10);



% [a, b, c] = pca(data);
% figure;
% hold on;
% grid on;
% plot3(b(classm, 1), b(classm, 2), b(classm, 3), '.');
% plot3(b(classf, 1), b(classf, 2), b(classf, 3), '.');
% plot3(b(classi, 1), b(classi, 2), b(classi, 3), '.');