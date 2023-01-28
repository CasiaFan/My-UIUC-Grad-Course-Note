% 8.3
a0=2.0;b0=10.0;a1=2.1;b1=10.1;
% a0=0.0;b0=20.0;a1=0.1;b1=20.1;
Ns=10:20:50;
a0s=[2.0 0.0];
b0s=[10.0 20.0];
a1s=[2.1 0.1];
b1s=[10.1 20.1];
num_figure = 3;
f1=figure('Position', [0 0 360*num_figure 360]);
f2=figure('Position', [0 0 360*num_figure 360]);
% f3=figure('Position', [0 0 360*num_figure 360]);
N_count = 1;
for N=Ns 
%     N=Ns;
%     a0=a0s(j);b0=b0s(j);a1=a1s(j);b1=b1s(j);
    g0=a0+(b0-a0)*rand(N,N,1000);
%     g1=a1+(b1-a1)*rand(N,N,1000);
    g1=a1+(b1-a1)*normrnd(0,2,[N,N,1000]);
    g0_mean = zeros(1000,1);
    g1_mean = zeros(1000,1);
    v0 = var(g0, 0, 'all');
    v1 = var(g1, 0, 'all');
    disp(v0);
    disp(v1);
    for i=1:1000
        g0_mean(i) = mean(g0(:,:,i), 'all');
        g1_mean(i) = mean(g1(:,:,i), 'all');
    end
    N_total=N*N;sigma=sqrt(v);u0=(a0+b0)/2;d=0.1;
    g0_mean_z = (g0_mean - u0)/(sigma/sqrt(N_total));
    g1_mean_z = (g1_mean - u0)/(sigma/sqrt(N_total));
    [N0,E0] = histcounts(g0_mean_z, "Normalization", 'probability');
    [N1,E1] = histcounts(g1_mean_z, "Normalization", 'probability');
    figure(f1);
    subplot(1, num_figure, N_count);
    histogram(g0_mean_z, E0, 'Normalization', 'probability'); 
    hold on; histogram(g1_mean_z, E1, 'Normalization', 'probability');
%     hold on; histfit(g0_mean_z);
%     hold on; histfit(g1_mean_z);
    set(gca, "FontSize", 16);
    xlabel("z (N="+num2str(N)+")", "Fontsize",20);
    ylabel("Pr", "Fontsize",20);
    
    count = 1;
    fpf = zeros(length(E0),1);
    tpf = zeros(length(E0),1);
    min_v = min(min(E0), min(E1));
    max_v = max(max(E0), max(E1));
    for t=E0
        fpf(count) = sum(N0(count:end));
        id = find(abs(E1-t)<1e-4);
        if isempty(id) 
            tpf(count)=sum(N1);
        else
            tpf(count)=sum(N1(id:end));
        end
        count = count+1;
    end
    figure(f2);
    subplot(1, num_figure, N_count);
    plot(fpf,tpf, 'Linewidth', 2);
    set(gca, "FontSize", 16);
    xlabel("FPF (N="+num2str(N)+")","Fontsize",20);
    ylabel("TPF", "Fontsize",20);
    bins=fpf(1:end-1)-fpf(2:end);
    centers = (tpf(1:end-1)+tpf(2:end))/2;
    auc = sum(bins.*centers);
    disp(sprintf("bin auc: %f", auc));
%     total_err = (1-tpf) + fpf;
%     [min_e, min_idx] = min(total_err);
%     figure(f1);
%     subplot(1, num_figure, N_count);
%     plot(E0, total_err, 'k-', 'Linewidth', 2, 'Color','blue');
%     disp(E0(min_idx))
    
    % use perfcurve to calculate auc and roc
%     labels = [zeros(1, length(g0_mean_z)) ones(1, length(g1_mean_z))];
%     scores = [g0_mean_z' g1_mean_z'];
%     [X,Y,T,AUC] = perfcurve(labels, scores, 1);
%     figure(f3);
%     subplot(1,num_figure,N_count);
%     plot(X,Y);
%     set(gca, "FontSize", 16);
%     xlabel("FPF (N="+num2str(N)+")","Fontsize",20);
%     ylabel("TPF", "Fontsize",20);
%     disp(sprintf('auc: %f', AUC));
    N_count = N_count + 1;
end

% 8.4
z=-5:0.1:10;Nz=length(z);
d=[0 0.1 0.5 1 2 5]; Nd=length(d);
Z0=normpdf(z,0,1);Z1=zeros(Nd,length(Z0));
TP=zeros(Nd,Nz);FP=zeros(Nd,Nz);AUC=zeros(size(d));
z1=-5:0.1:5;
Z2=normpdf(z1,0,1);Z3=normpdf(z1,0,2);Z4=normpdf(z1,0,0.5);
figure;plot(z1,Z2);hold on;
plot(z1, Z3, 'color', 'red', 'linewidth',2); hold on;
plot(z1, Z4, 'color', 'blue', 'linewidth',2);
figure('position',[0 0 1080 360]);
subplot(1,3,1);
plot(z,Z0,'k', 'linewidth',2);hold on;
for j=1:Nd
    Z1(j,:)=normpdf(z,d(j),2);
    if(j~=1);subplot(1,3,1);plot(z,Z1(j,:),'k','linewidth', 2);end
    for k=Nz:-1:1
        TP(j,Nz-k+1)=cdf('norm',z(k),d(j),2, 'upper');
        FP(j,Nz-k+1)=cdf('norm',z(k),0,1,'upper');
    end
    AUC(j)=trapz(FP(j,:),TP(j,:));       
end
xlabel('z');axis square;
for j=1:Nd
    subplot(1,3,2)
    if j==1
        plot(FP(j,:),TP(j,:),'k*','linewidth',2);hold on;
    else
        plot(FP(j,:),TP(j,:),'k','linewidth',2);hold on;
    end
end
xlabel('FPF or \alpha');ylabel('TPF or 1-\beta');axis square;hold off;
subplot(1,3,3)
AUC_o = [0.5000    0.5282    0.6381    0.7601    0.9212    1];
plot(d,AUC_o,'k--','linewidth',2,'color','blue');hold on;
plot(d,AUC,'k-','linewidth',2);
xlabel('d');axis square;

% 8.5
samp=10;exp=10;
g0=normrnd(220,20,[samp,1,exp]);
ci_l = mean(g0, 'all')-0.754*sqrt(var(g0,0,'all'));
ci_h = mean(g0, 'all')+0.754*sqrt(var(g0,0,'all'));
disp(ci_l);disp(ci_h);disp(sqrt(var(g0,0,'all')));
% ci_l = 204.9; ci_h=235.1;
g0_mean = zeros(exp,1);
% g0_var = zeros(exp,1);
for i=1:exp
    g0_mean(i) = mean(g0(:,:,i), 'all');
%     g0_var(i) = var(g0(:,:,i), 0, 'all');
end
% [N,E]=histcounts(g0_mean, 5, "Normalization", 'probability');
% figure;
% histogram(g0_mean,E,'Normalization', 'probability');
match_count = sum((ci_l < g0_mean) & (g0_mean< ci_h));
disp(match_count)
figure('position',[0 0 480 120]);
plot(g0_mean, zeros(1,length(g0_mean)), "o");xlabel("mean");xlim([ci_l-3,ci_h+3]);hold on;
plot([ci_l ci_h],[0 0], "or");line([ci_l-3,ci_h+3],[0,0], 'color', 'black');set(gca, 'ytick',[])
x=1:10;y=[10 10 10 9 9 10 10 10 10 10];
figure('position',[0 0 480 160]);plot(x,y,'ko-','linewidth',2);
xlabel("trials", 'fontsize', 16);ylabel("#mean falls in CI",'fontsize', 16);ylim([8,10])