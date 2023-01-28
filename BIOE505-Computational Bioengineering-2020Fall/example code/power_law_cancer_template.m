% load TP53_mutations_COSMIC.mat;
% load APC_mutations_COSMIC.mat;
load BCL2_mutations_COSMIC.mat;
[a,b]=hist(Count,1:max(Count));
n1=b; p1=a./sum(a);
figure; subplot(1,2,1); loglog(n1, p1,'ko');
hold on; axis manual;
subplot(1,2,1); loglog(n1, n1.^-2,'r--');
max1=max(Count);
clear cdf1
for m=1:max1;
    cdf1(m)=sum(p1(m:end));
end;
subplot(1,2,2);loglog(n1, cdf1,'ko');
hold on; axis manual;
subplot(1,2,2); loglog(n1, n1.^-1,'r--');