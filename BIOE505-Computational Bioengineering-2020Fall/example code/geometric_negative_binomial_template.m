%% Geometric 
Stats=100000;
p=0.1;
r2=random('Geometric',p, Stats, 1);
r2=r2+1; % correct for Matlab counting only failures among trials
disp(mean(r2));
disp(var(r2));
disp(std(r2));
[a,b]=hist(r2, 1:max(r2));
p_g=a./sum(a);
figure; semilogy(b,p_g,'ko-');
%% Negative-binomial
Stats=100000;
r=3; p=0.1;
r2=random('Negative Binomial', r, p, Stats, 1);
r2=r2+1; % correct for Matlab counting only failures among trials
disp(mean(r2));
disp(var(r2));
disp(std(r2));
[a,b]=hist(r2, 1:max(r2));
p_nb=a./sum(a);
figure; semilogy(b,p_nb,'ko-');
%%