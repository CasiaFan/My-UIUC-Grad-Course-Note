% 9.3
clear all;Nx=[10 100 1000 10000];
figure('Position', [0,0,1920, 960]);
set(gca, 'OUterPosition', [0, 0, 1, 1])
count=1;
for N=Nx
    scaffold=1/N;
    mx1=0;my1=0;sx1=40;sy1=25;r1=-0.7;
    mx2=0;my2=40;sx2=40;sy2=25;r2=-0.7;
    % mx1=500;my1=150;sx1=120;sy1=40;r1=0.3;
    % mx2=170;my2=400;sx2=120;sy2=250;r2=0.7;
    m1=ones(N,2);m1(:,1)=m1(:,1)*mx1;m1(:,2)=m1(:,2)*my1;
    m2=ones(N,2);m2(:,1)=m2(:,1)*mx2;m2(:,2)=m2(:,2)*my2;
    M1=[mx1;my1];M2=[mx2;my2];
    K1=[sx1^2 r1*sy1*sx1;r1*sx1*sy1 sy1^2];
    K2=[sx2^2 r2*sy2*sx2;r2*sx2*sy2 sy2^2];
    Swi=2*eye(2)/(K1+K2);
    X1=mvnrnd(m1,K1);X2=mvnrnd(m2,K2);
%     figure('Position', [0 0 1280 640]);
    subplot(2,4,count);
    plot(X1(:,1),X1(:,2),'r.');axis square;hold on;
    plot(X2(:,1),X2(:,2),'b^');
    t=-60:2:20;Nt=length(t);p=-150:150;TPF=zeros(1,Nt);FPF=TPF;
    for i=1:Nt
        a=2*(M2-M1)'*Swi;
        b=t(i)+M2'*Swi*M2-M1'*Swi*M1;
        q=(b-a(1)*p)/a(2);
        C1=0;C2=0;
        for j=1:N
            if (X1(j,2)>(b-a(1)*X1(j,1))/a(2))
                C1=C1+1;
            end
            if (X2(j,2)>(b-a(1)*X2(j,1))/a(2))
                C2=C2+1;
            end
        end
        TPF(i)=scaffold*C2;FPF(i)=scaffold*C1;
        if(t(i)==-20);plot(p,q,'r--'); text(-140,-20, "t=-20", 'fontsize', 16);end
        if(t(i)==0);plot(p,q,'k--', 'linewidth',2);text(-140, 100, "t=0", 'fontsize', 16);end
        if(t(i)==20);plot(p,q,'b--'); text(-80, 140, "t=20", 'fontsize', 16); hold off; end
    end
    xlim([-150 150]);ylim([-100 150]);
    xlabel("$x_1$(N="+num2str(N)+")","interpreter", 'latex', 'fontsize', 20);
    ylabel('$x_2$','interpreter','latex', 'fontsize',20);
    set(gca, 'Fontsize', 16);
    subplot(2,4,count+4);
    plot(FPF,TPF,'k','linewidth',2)
    AUC=trapz(FPF,TPF);
    axis square;
    xlabel("FPF(N="+num2str(N)+")", 'fontsize', 20);
    ylabel('TPF','fontsize',20);
    set(gca, 'Fontsize', 16);
    text(0.7,0.9,"AUC="+num2str(-AUC), 'fontsize', 16);

    mu1=[mx1;my1];mu2=[mx2;my2];
    mu_b=(mu1+mu2)/2;
    Sb=0.5*(mu1-mu_b)*(mu1-mu_b)'+0.5*(mu2-mu_b)*(mu2-mu_b)';
    Jh=trace(Swi*Sb)

    Bmu=(mu2-mu1)'*Swi*(mu2-mu1)/8
    count=count+1;
end

% 9.4 
clear all;N=1000;
scaffold=1/N;
mx1=0;my1=0;mz1=-10;sx1=40;sy1=40;sz1=10;r12=0;r13=0.4;r23=0.4;
mx2=0;my2=40;mz2=15;sx2=40;sy2=40;sz2=10;
m1=ones(N,2);m1(:,1)=m1(:,1)*mx1;m1(:,2)=m1(:,2)*my1;m1(:,3)=mz1;
m2=ones(N,2);m2(:,1)=m2(:,1)*mx2;m2(:,2)=m2(:,2)*my2;m2(:,3)=mz2;
M1=[mx1;my1;mz1];M2=[mx2;my2;mz2];
K1=[sx1^2 r12*sy1*sx1 r13*sx1*sz1;r12*sx1*sy1 sy1^2 r23*sy1*sz1; r13*sx1*sz1 r23*sy1*sz1 sz1^2];
K2=K1;
Swi=2*eye(3)/(K1+K2);
X1=mvnrnd(m1,K1);X2=mvnrnd(m2,K2);
figure('Position', [0 0 1280 640]);
subplot(1,2,1);
scatter3(X1(:,1),X1(:,2),X1(:,3), 'r');axis square; hold on;
scatter3(X2(:,1),X2(:,2),X2(:,3), 'b');axis square; hold on;
t=-60:2:20;Nt=length(t);TPF=zeros(1,Nt);FPF=TPF;
px=[-150 150 150 -150];py=[-150 -150 150 150];
for i=1:Nt
    a=2*(M2-M1)'*Swi;
    b=t(i)+M2'*Swi*M2-M1'*Swi*M1;
    pz=(b-a(1)*px-a(2)*py)/a(3);
    C1=0;C2=0;
    for j=1:N
        if (X1(j,3)>(b-a(1)*X1(j,1)-a(2)*X1(j,2))/a(3))
            C1=C1+1;
        end
        if (X2(j,3)>(b-a(1)*X2(j,1)-a(2)*X2(j,2))/a(3))
            C2=C2+1;
        end
    end
    TPF(i)=scaffold*C2;FPF(i)=scaffold*C1;
    if(t(i)==0);patch(px,py,pz, 'k','faceAlpha', 0.2);text(150, 150, 25, 't=0', 'fontsize', 16);end
end
set(gca, 'Fontsize', 11);
xlabel('x','fontsize', 20);
ylabel('y','fontsize', 20);
zlabel('z','fontsize', 20);
subplot(1,2,2);
plot(FPF,TPF,'k','linewidth',2);axis square;
AUC=trapz(FPF,TPF);
xlabel("FPF(N="+num2str(N)+")", 'fontsize', 20);
ylabel('TPF','fontsize',20);
set(gca, 'Fontsize', 16);
text(0.7,0.9,"AUC="+num2str(-AUC), 'fontsize', 16);
% compute J
mu_ave=(M1+M2)/2;
Sb=0.5*((M1-mu_ave)*(M1-mu_ave)'+(M2-mu_ave)*(M2-mu_ave)');
Jh=trace(Swi*Sb);
% comput B
Bmu=(M2-M1)'*Swi*(M2-M1)/8;
Bk=0.5*log(det((K1+K2)/2)/sqrt(det(K1)*det(K2)));
B=Bmu+Bk;

% 9.5
rng('default');
a=4;N=200;X1p=3*randn(N,2)+2*ones(N,2);
X2=2*randn(20,2)-ones(20,2);
N1=0;X1=zeros(length(N),2);
for j=1:N
    r=sqrt(X1p(j,1)^2+X1p(j,2)^2);
    if r>a
        N1=N1+1;
        X1(N1,:)=X1p(j,:);
    end
end
figure('Position', [0 0 1280 640]);
subplot(1,2,1);
c1x1=mean(X1(:, 1));c1x2=mean(X1(:,2));
c2x1=mean(X2(:, 1));c2x2=mean(X2(:,2));
cx1=[c1x1 c2x1]; cx2=[c1x2 c2x2];
X=[X1;X2];plot(X(:,1),X(:,2),'ko','MarkerSize',10); hold on;
plot(X2(:,1),X2(:,2),'r.',"MarkerSize", 30);
plot(cx1, cx2, 'kx','markersize',12,'linewidth',3); hold off
xlabel('x_1');ylabel('x_2')
set(gca, 'fontsize',16);
axis square;

options=statset('Display', 'final');
[idx, C] = kmeans(X, 2, 'replicates', 3, 'options', options);
ic =0;
for j=1:length(X1)
    if idx(j) ~= 1;ic=ic+1;end
end
for k=length(X1)+1:length(X)
    if idx(k) ~= 2;ic=ic+1;end
end
subplot(1,2,2);
plot(X(idx==1,1), X(idx==1,2),'ro', 'markersize',6);hold on;
plot(X(idx==2,1),X(idx==2,2),'b+', 'markersize',6);
plot(C(:,1),C(:,2),'kx','markersize',12,'linewidth',3);hold off
xlabel("x_1");ylabel("y_1");
axis square;
set(gca, 'fontsize',16);

    