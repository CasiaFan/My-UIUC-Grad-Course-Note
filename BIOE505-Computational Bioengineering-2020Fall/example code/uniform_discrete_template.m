b=10; a=1; % b= upper bound; a= lower bound (inclusive)'
Stats=100000; % sample size to generate
r1=rand(Stats,1); 
% r2=floor(10*r1)+1; 
r2 = randi(10, Stats, 1);
mean(r2)
var(r2)
std(r2)
[hy,hx]=hist(r2, 1:10); % hist generates histogram in bins 1,2,3...,10
% hy - number of counts in each bin; hx - coordinates of bins
p_f=hy./sum(hy); % normalize counts to add up to 1
figure; plot(hx,p_f, 'ko-'); ylim([0, max(p_f)+0.01]); % plot the PMF
hold on; histogram(r2, 'normalization', 'pdf');





