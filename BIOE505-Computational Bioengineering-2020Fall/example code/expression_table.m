% corr
g1=2872;g2=1269;
r1=381;r2=2741;
ge1=exp_t(g1, :);ge2=exp_t(g2, :);
lm1=fitlm(ge1,ge2);
re1=exp_t(r1, :);re2=exp_t(r2, :);
lm2=fitlm(re1, re2);

% 1
int1=1.9239;slop1=0.83851;
y1=slop1*ge1+int1;
y_mean =mean(ge2);
r_sq = 1-sum((ge2-y1).^2)/sum((ge2-y_mean).^2);
disp(r_sq);

%2
int2=7.9282;slop2=0.024361;
y2=slop2*re1+int2;
y_mean2 =mean(re2);
r_sq2 = 1-sum((re2-y2).^2)/sum((re2-y_mean2).^2);
disp(r_sq2);
disp("2872");disp(gene_description(2872));
disp("1269");disp(gene_description(1269));
disp("381");disp(gene_description(381));
disp("2741");disp(gene_description(2741));