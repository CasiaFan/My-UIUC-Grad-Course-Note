close all; dt=0.01;T0=10;t=0:dt:T0;N=length(t);
f=ones(size(t));
f(51:100)=-1;f(151:200)=-1;f(251:300)=-1;f(351:400)=-1;
f(426:450)=-1;f(476:500)=-1;f(526:550)=-1;f(576:600)=-1;
f(612:625)=-1;f(637:650)=-1;f(662:675)=-1;f(687:700)=-1;
f(712:725)=-1;f(737:750)=-1;f(762:775)=-1;f(787:800)=-1;
f(806:812)=-1;f(818:824)=-1;f(830:836)=-1;f(842:848)=-1;
f(854:860)=-1;f(866:872)=-1;f(878:884)=-1;f(890:896)=-1;
f(902:908)=-1;f(914:920)=-1;f(926:932)=-1;f(938:944)=-1;
f(950:956)=-1;f(962:968)=-1;f(974:980)=-1;f(986:992)=-1;
% figure('Position', [0, 0, 480, 240]);plot(t,f,'k','linewidth', 1); hold on;
% yline(0, '--');
% xlabel('time (s)');ylabel('f(t)');ylim([-1.5, 1.5]); hold off;
% set(gca, 'fontsize', 16);
% e=0.01*randn(1,1001);SNR=10*log10(sum(f.*f)/0.01^2);
% title(['SNR=', num2str(SNR)]);
% figure('Position', [0, 0, 480, 640]);
mp = colormap(winter);
cint=length(mp)/4;
colors=[mp(1,:); mp(1+cint,:); mp(1+cint*2, :); mp(1+cint*3,:)];
% subplot(3,2,1);plot(t,f,'k','linewidth', 0.5);
% xlabel('time (s)');ylabel('f(t)');
e=0.01*randn(1,1001);SNR=10*log10(sum(f.*f)/0.01^2);
% title(['SNR=', num2str(SNR)]);
zeta=0.3;Omega0=2*pi*15;fac=sqrt(1-zeta^2);
h=exp(-Omega0*zeta*t).*sin(Omega0*fac*t)/(Omega0*fac); % impulse response
h=h/sqrt(sum(h.*h)*dt); % normalize
intv1=1;intv2=51;
% subplot(2,1,1);plot(t(intv1:intv2),h(intv1:intv2), 'k', 'linewidth', 2);
% xlabel('time (s)');ylabel('h(t)');
% yline(0, '--');
% title("Impulse response of sensor 1")
% set(gca, 'fontsize', 16)

gp=conv(f,h)*dt;g=gp(1:1001); % measure gt=[h*f](t)
pred=8*(g+e);
cap=[1 400 600 800 1000];
figure('Position', [0,0,480,1280]);
for i=1:4
    subplot(4,1,i);
    plot(t(cap(i):cap(i+1)), pred(cap(i):cap(i+1)), 'k--', 'linewidth', 2, 'color', 'blue'); hold on
    if i==1
        xlim([0, 4])
    else
        xlim([2*i, 2*i+2])
    end
    ylim([-1.5, 1.5])
    yline(0, '--')
    xlabel('time (s)');ylabel('g(t)');
end
MSE1=(f(cap1:cap2)-pred(cap1:cap2)).^2;
% title(['MSE1=',num2str(MSE1)]);
u=0:1/T0:1/(2*dt);
Fp=fft(f);F=20*log10(abs(Fp(1:501)));
Hp=fft(h);H=20*log10(abs(Hp(1:501)));
Gp=fft(g);G=20*log10(abs(Gp(1:501)));
% subplot(1,3,1);
% plot(u, F, 'k', 'linewidth', 1); hold on;
% plot(u, H, 'k', 'linewidth', 2, 'color', 'blue'); hold off
% plot(u, G, 'k', 'linewidth', 2, 'color', 'green');hold off;
% xlabel('frequencu (Hz)');ylabel('frequency spectrum (dB)')
% xlim([0,50])
% set(gca, 'fontsize', 16)
% title("Frequency spectrum of sensor 1's impulse response");
% subplot(1,3,3);
% plot(u, H, 'k', 'linewidth', 2, 'color', 'blue'); hold on;
% xlabel('frequencu (Hz)');ylabel('frequency spectrum (dB)');
Om=2*pi*[4.2,9.4,14.4,23.7];w=[1;2;4;8];
ze=0.2;fac=sqrt(1-ze^2);
for j=1:length(Om)
    h(j,:)=exp(-Om(j)*ze*t).*sin(Om(j)*fac*t)/(Om(j)*fac);  % impulse response
    h(j,:)=h(j,:)/sqrt(sum(h(j,:).*h(j,:))*dt);
end
hhh=h'*w; hh=hhh';
hhn=10^(-18.3/20)*hhh'; % response functions into one and attenuate so peak matches peak of single sensor device
% figure(1);subplot(2,1,2);
% for i=1:4
%     plot(t(intv1:intv2), h(i, intv1:intv2), 'k--', 'color', colors(i,:), 'linewidth', 2); hold on
% end
% xlabel('time (s)');ylabel('h(t)');
% subplot(2,1,2);plot(t(intv1:intv2), hh(intv1:intv2), 'k', 'linewidth', 2); hold off
% yline(0, '--')
% xlabel('time(s)');ylabel('h(t)');
% title('impulse response of sensor 2')
% set(gca, 'fontsize', 16)
Hp=fft(hh);H=20*log10(abs(Hp(1:501)));
Hpn=fft(hhn);Hn=20*log10(abs(Hpn(1:501)));
% plot(u, Hn, 'k', 'linewidth', 2, 'color', 'red'); hold off;
% xlabel('frequencu (Hz)');ylabel('frequency spectrum (dB)');
% xlim([0,50])
% set(gca, 'fontsize',16);
% title("comparison of frequency spectrum of sensor 1 and 2 with normalized peak value")
gp=conv(f,hhn)*dt;g=gp(1:1001);
pred=5*(g+e);
MSE2=(f(cap1:cap2)-pred(cap1:cap2)).^2;
for i=1:4
    subplot(4,1,i);
    plot(t(cap(i):cap(i+1)),pred(cap(i):cap(i+1)),'k--','linewidth',2, 'color', 'red'); hold off;
    title(['MSE1=', num2str(mean(MSE1(cap(i):cap(i+1)))), ';MSE2=',num2str(mean(MSE2(cap(i):cap(i+1))))]);
    if i==1
        xlim([0, 4])
    else
        xlim([2*i, 2*i+2])
    end
    set(gca, 'fontsize',16)
end

for j=1:length(Om)
    Hf(j,:)=fft(h(j,:));HHf(j,:)=20*log10(w(j)*abs(Hf(j,1:501)));
%     plot(u, HHf(j,:), 'k', 'linewidth', 1, 'color', colors(j,:)); hold on;
end
% plot(u, H, 'k', 'linewidth', 2, 'color', 'red');hold off; 
% xlabel('frequencu (Hz)');ylabel('frequency spectrum (dB)');
% xlim([0,50])
% set(gca, 'fontsize',16)
% 
% syms omega zeta t s;
% h=exp(-omega*zeta*t)*sin(omega*sqrt(1-zeta^2)*t)/(omega*sqrt(1-zeta^2));
% H=laplace(h, t, s);
% omega=2*pi*15;zeta=0.3;
% HH=tf([0,0,0],[1,2*omega*zeta,-omega^2]);
% pzmap(HH)