function ml = mlclass(scores)

N = height(scores);
ml = char(zeros(N,1));
for i = 1:N
    mlclass = [scores.M(i) scores.F(i) scores.I(i)];

    if scores.M(i) == max(mlclass)
        ml(i) = 'M';
    end
    if scores.F(i) == max(mlclass)
        ml(i) = 'F';
    end
    if scores.I(i) == max(mlclass)
        ml(i) = 'I';
    end
end