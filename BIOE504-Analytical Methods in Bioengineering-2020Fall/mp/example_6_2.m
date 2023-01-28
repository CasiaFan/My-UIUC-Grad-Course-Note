% dt=0.01;t=-100:dt:100;
% x=1/sqrt(2*pi);s=10*dt;a=100;
% T=a*dt;
% h0=x/s*exp(-t.^2/(2*s^2));
% lp=21;lp2=2*lp-1;
% plot(t,h0);hold on;
% hh=zeros(1,lp);
% for m=1:lp
%     h1=x/s*exp(-(t-(m-1)*T).^2/(2*s^2));plot(t, h1, 'k');
%     hh(m) = h1*h0'/(h0*h0');
% end;hold off;
% h(1:lp)=hh;h(lp+1:lp2)=hh(lp:-1:2);
% H=zeros(lp2);H(1,:)=h;
% for j=2:lp2
%     H(j,1:j-1)=h(lp2-j+2:lp2);
%     H(j,j:lp2)=h(1:lp2-j+1);
% end;figure;imagesc(H);colormap(gray);axis square
% [U,D]=eig(H);d=diag(D);
% figure;stem(d(lp2:-1:1),'k')
% 
% b=(cos(2*pi*t/5).^2).*exp(-t.^2/(2*3^2))+exp(-(t-20).^2/(x*3^2));
% figure;plot(t,b);
% f=exp(-t.^2/(2*3^2)).*cos(2*pi*t/5).^2;
% pff=xcorr(f,f);ppf=pff(10001:100:30001);P=zeros(lp2);
% P(1,1:21)=ppf(101:121);P(1,22:41)=ppf(81:100);plot(P(1,:))
% for j=2:lp2
%     P(j,1:j-1)=P(1,lp2-j+2:lp2);
%     P(j,j:lp2)=P(1,1:lp2-j+1);
% end;figure;imagesc(P);colormap(gray);axis square
% DD=U'*P*U;dd=diag(DD);
% figure;stem(dd(lp2:-1:1), 'k')
% diag(DD)'*diag(D)

hz=100;t=0:1/hz:1;uu=0:hz;a=2;
noi=randn(length(t),1)*a;
gu = fft(noi)/hz;
figure("Position", [0 0 450 600]);subplot(3,1,1);
plot(uu, real(gu));
ylim([-1,1]);
xlabel("u");
ylabel("$\mathcal{F}e$", "interpreter", "latex", "fontsize", 20);
set(gca, "fontsize", 16);
text(2, 0.8, "k=100", "fontsize", 20);

subplot(3,1,2);
hz=1000;t=0:1/hz:1;uu=0:hz;
noi=randn(length(t),1)*a;
gu = fft(noi)/hz;
plot(uu, real(gu));
ylim([-1,1]);
xlabel("u");
ylabel("$\mathcal{F}e$", "interpreter", "latex", "fontsize", 20);
set(gca, "fontsize", 16);
text(20, 0.8, "k=1000", "fontsize", 20);

subplot(3,1,3);
hz=5000;t=0:1/hz:1;uu=0:hz;
noi=randn(length(t),1)*a;
gu = fft(noi)/hz;
plot(uu, real(gu));
ylim([-1,1]);
xlabel("u");
ylabel("$\mathcal{F}e$", "interpreter", "latex", "fontsize", 20);
set(gca, "fontsize", 16);
text(20, 0.8, "k=5000", "fontsize", 20);


