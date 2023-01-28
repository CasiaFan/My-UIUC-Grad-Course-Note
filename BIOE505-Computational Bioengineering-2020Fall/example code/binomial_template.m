%% Binomial distribution
n=200; p=0.2;
Stats=100000;
r1=rand(Stats,n)  <  p;
r2=sum(r1, 2); 
%%
disp('Sample mean');
disp(mean(r2));
disp('Expected mean:')
disp(n.*p);
disp('Sample variance');
disp(var(r2));
disp('Expected variance:')
disp(n.*p.*(1-p));
%%
[a,b]=hist(r2, 0:1:n);
p_b=a./sum(a);

%%
figure; stem(b,p_b);
%%
figure; semilogy(b,p_b,'ko-') 