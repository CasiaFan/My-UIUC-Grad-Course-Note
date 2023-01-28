load cancer_wdbc.mat
X=cancerwdbc(:,2:end);
corr_mat=corr(X);
cancer_corr=corr(cancer, X);
cancer_corr=cancer_corr';
for m=1:30;
disp([num2str(m),',', num2str(round(cancer_corr(m), 3)), ',', feature_names{m}]);
end;
figure; plot(cancer_corr, 'ko-');
a4=gca;
set(a4,'XTick',1:30);
set(a4,'XTickLabel',feature_names);
set(a4,'XTickLabelRotation', 270);
%%
[i,j,c]=find(corr_mat);
c1=c(i>j); whos c1;
figure; histogram(c1);
%%
aux=corr_mat;
aux(end+1,:)=aux(end,:);
aux(:,end+1)=aux(:,end);
figure; pcolor(aux); 
colormap jet; 
colorbar;
a4=gca;
set(a4,'XTick',1.5:30+0.5);
set(a4,'XTickLabel',feature_names);
set(a4,'XTickLabelRotation', 270);
set(a4,'YTick',1.5:30+0.5);
set(a4,'YTickLabel',feature_names);
%%
Z=zscore(X);
[v,d]=eigs(corr_mat,30);
ev_d=diag(d);
disp(ev_d(1:10)./30); % shows fraction explained by each p.c.
%%
% columns of v are eigenvectors
figure; 
plot(v(:,1),'bo-');
a4=gca;
set(a4,'XTick',1:30);
set(a4,'XTickLabel',feature_names);
set(a4,'XTickLabelRotation', 270);
%%
% plot 1-st and 2-nd eigenvalues against each other for 
% normal (red) and cancer (blue) patients
scores=Z*v;
m=1; n=2; 
figure; 
plot(scores(cancer==0,m), scores(cancer==0,n), 'gd');
hold on;
plot(scores(cancer==1,m), scores(cancer==1,n), 'ro');
legend('normal','cancer');
%%
% We will revisit this problem when discussion regression
% the only statistically significant linear fits to cancer/healthy 
% binary variable is x6 ('compactness' or perimeter^2/area) 
% and x19 'symmetry std' where symmetry is the difference in dimensions perpendicular to the longest diameter)
% predictability is low: R-squared: 0.111,  Adjusted R-Squared: 0.0619
% corresponding to Pearson correlation = sqrt(R) around 0.24
%
lm=fitlm(X,cancer)
%
% Linear regression model:
%     y ~ [Linear formula with 31 terms in 30 predictors]
% 
% Estimated Coefficients:
%                     Estimate         SE          tStat       pValue 
%                    ___________    _________    _________    ________
% 
%     (Intercept)        0.34385      0.84928      0.40487     0.68573
%     x1                 0.39482      0.34429       1.1468     0.25199
%     x2                0.022138     0.015763       1.4045     0.16076
%     x3               -0.065295     0.049802      -1.3111     0.19039
%     x4               0.0010708    0.0010423       1.0273     0.30475
%     x5                 -3.2567       4.0028      -0.8136     0.41623
%     x6                  5.8996       2.6464       2.2293    0.026205
%     x7                 -3.9316       2.0753      -1.8945    0.058694
%     x8                 0.30822       3.9272     0.078484     0.93747
%     x9                -0.34419       1.4738     -0.23354     0.81543
%     x10                 16.616       11.057       1.5027     0.13349
%     x11              -0.068126      0.61605     -0.11058     0.91199
%     x12               0.057443     0.073108      0.78573     0.43237
%     x13              -0.049652      0.08159     -0.60856     0.54307
%     x14              0.0031701    0.0027737       1.1429     0.25357
%     x15                  5.187       13.146      0.39458     0.69331
%     x16                -3.2215       4.3048     -0.74835     0.45458
%     x17                 0.7095        2.581      0.27489     0.78351
%     x18               -0.79077       10.818    -0.073096     0.94176
%     x19                -12.647       5.4123      -2.3367    0.019822
%     x20                0.54414       23.169     0.023486     0.98127
%     x21               -0.16311      0.11502      -1.4181     0.15674
%     x22             -0.0051146     0.013789     -0.37093     0.71083
%     x23               0.020302     0.011779       1.7236    0.085348
%     x24            -0.00026401    0.0006343     -0.41623     0.67741
%     x25                -1.3677       2.8467     -0.48044     0.63111
%     x26                -0.6877      0.76014      -0.9047     0.36603
%     x27                0.97283      0.53301       1.8252    0.068532
%     x28                 0.4555       1.8141      0.25109     0.80184
%     x29              -0.062031      0.98083    -0.063243      0.9496
%     x30                -5.5025       4.7289      -1.1636     0.24511
% 
% 
% Number of observations: 569, Error degrees of freedom: 538
% Root Mean Squared Error: 0.469
% R-squared: 0.111,  Adjusted R-Squared: 0.0619
% F-statistic vs. constant model: 2.25, p-value = 0.000207