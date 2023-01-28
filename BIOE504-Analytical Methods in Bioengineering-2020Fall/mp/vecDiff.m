function res = vecDiff(x)
    a = [1 1 2 2; 1 3 3 2; 3 1 3 2; 2 6 6 4];
    b = randn(4) * x;
    ae = eig(a);
    be = eig(b+a);
    res = mean(ae-be);
end