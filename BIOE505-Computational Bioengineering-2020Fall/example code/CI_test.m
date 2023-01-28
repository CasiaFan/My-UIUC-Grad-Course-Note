labs=1000;n=20;mu=3;sig=2;
data = sig*randn(labs, n)+mu;

counter=0;
xbar=zeros(labs,1);
sbar=zeros(labs,1);
for i=1:labs
    xmean = mean(data(i,:));
    if xmean > mu+1.96*sig/sqrt(n) || xmean < mu-1.96*sig/sqrt(n)
        counter = counter + 1;
    end
end
disp(counter)
    