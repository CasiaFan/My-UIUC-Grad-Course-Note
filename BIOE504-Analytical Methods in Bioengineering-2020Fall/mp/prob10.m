% 10.2
% V1=1;as=[10 1.001 0.1];
% t0=-1:0.01:-0.01;t1=0:0.01:6;t=-1:0.01:6;
% c1_0=ones(1, length(t0));
% c2_0=zeros(1, length(t0));
% c1 = exp(-t1);
% c1_z = cat(2, c1_0, c1);
% figure;
% plot(t, c1_z, 'k-', 'linewidth', 2); hold on;
% c_set=['b-' 'r-' 'g-'];
% for i=1:length(as)
%     c2 = 1/(1/as(i)-1)*(exp(-as(i)*t1)-exp(-t1));
%     c2_z = cat(2, c2_0, c2);
%     plot(t, c2_z, c_set(i), 'linewidth', 2); 
% end
% xlabel("y=qt/V_1")
% ylabel("c_i/c_0")
% set(gca, "fontsize", 16)

% 10.4
% dt=0.1;
% t0=-5:dt:0;t1=dt:dt:168;t=-5:dt:168;
% lam_m=log(2)/66;lam_T=log(2)/6;
% c1_0=ones(1, length(t0))*lam_m;
% c2_0=zeros(1, length(t0));
% N0M=1;
% c1 = lam_m*exp(-lam_m*t1)*N0M;
% c1_z = cat(2, c1_0, c1);
% figure;
% plot(t, c1_z, 'k-', 'linewidth', 2); hold on;
% N0T = 0;
% Tmilk=round(log(lam_m/lam_T)/(lam_m-lam_T), 1);
% n_milk=0;
% total_a=0;rec_a=0;
% total_NT=0;
% for ti=t1
%     c2 = lam_T*lam_m*N0M/(lam_T-lam_m)*(exp(-lam_m*(ti-Tmilk*n_milk))-exp(-lam_T*(ti-Tmilk*n_milk)))+N0T;
%     c2_0(end+1)=c2;
%     total_a=dt*c2+total_a;
%     if mod(ti, Tmilk) == 0
%         n_milk=n_milk+1;
%         N0M=exp(-lam_m*ti);
%         N0T=0.1*c2;
%         NT=lam_m*N0M/(lam_T-lam_m)*(exp(-lam_m*(ti-Tmilk*n_milk))-exp(-lam_T*(ti-Tmilk*n_milk)))+N0T/lam_T;
%         rec_a = rec_a+0.9*c2;
%         total_NT=total_NT+NT;
%     end
% end
% plot(t, c2_0, "r-", 'linewidth', 2); 
% xlabel("t")
% ylabel("Activity")
% set(gca, "fontsize", 16)

% 10.5
% lynx=[4.0 6.1 9.8 35.2 59.4 41.7 19.0 13.0 8.3 9.1 7.4 8.0 12.3 19.5 45.7 51.1 29.7 15.8 9.7 10.1 8.6];
% rab=[30.0 47.2 70.2 77.4 36.3 20.6 18.1 21.4 22.0 25.4 27.1 40.3 57.0 76.6 52.3 19.5 11.2 7.6 14.6 16.2 24.7];
% min_disc = 1e10;
% for k=0.1:0.02:0.7
%     for alpha=0.01:0.002:0.03
%         for gamma=0.8:0.05:1
%             for beta=0.1:0.02:0.7
% %                 k=0.5;alpha=0.02;
% %                 gamma=1;beta=0.6;
%                 theta=[k alpha;gamma beta];
%                 Ns=[k/alpha;beta/(alpha*gamma)];
%                 N0=[4;30];
%                 trange=[0 20];
%                 [tt,N]=ode45(@(tt,N) lotka1(tt, N, theta), trange, N0);
%                 xi=[N(:,1)-Ns(1),N(:,2)-Ns(2)];
% %                 subplot(1,3,1);
% %                 plot(tt,xi);title('relative populations');
% %                 % xlabel('time (yrs)');legend('\xi_1', 'xi_2');
% %                 subplot(1,3,2);plot(tt,N)
% %                 title('Predator & Prey Populations'); xlabel('time (yrs)');
%                 % ylabel('Number in Population x 10^3');legend('Predator (M)', 'Prey (N)', 'Location', 'North')
% %                 subplot(1,3,3);plot(N(:, 1), N(:,2)); hold on
%                 % title('Phase Plane Plot'); xlabel('Predator Population x 10^3');ylabel('Prey Population x 10^3');
% %                 hold on; plot(Ns(1), Ns(2), "+");
% %                 x1min=min(N(:, 1)-Ns(1));
%                 x1max=max(N(:, 1)-Ns(1));x2max=max(N(:,2)-Ns(2));
%                 % dx1=(x1max-x1min)/10;x1=x1min:dx1:x1max;
%                 a=x2max^2/Ns(2);
%                 b=x1max^2/gamma/Ns(1);
%                 x1_l=lynx-Ns(1);
%                 x2_l=real(sqrt((a-x1_l.^2/gamma/Ns(1))*Ns(2)));
%                 x2_r = rab - Ns(2);
%                 x1_r = real(sqrt((b-x2_r.^2/Ns(2))*gamma*Ns(1)));
%                 rab_disc = abs(rab-Ns(2));
%                 pred_rab_disc = x2_l-rab_disc;
%                 % mse for predict rabit
%                 rab_d = sqrt(sum((pred_rab_disc - rab_disc).^2)/length(rab));
%                 lynx_disc = abs(lynx - Ns(1));
%                 pred_lynx_disc = x1_r - lynx_disc;
%                 % mse for predict lynx
%                 lynx_d = sqrt(sum((pred_lynx_disc-lynx_disc).^2)/length(lynx));
% %                 rab_d = 0;
%                 disc = lynx_d + rab_d;
%                 if disc < min_disc
%                     min_disc = disc;
%                     disp("dic value: ")
%                     disp(disc)
%                     disp("k")
%                     disp(k)
%                     disp("alpha")
%                     disp(alpha)
%                     disp("gamma")
%                     disp(gamma)
%                     disp("beta")
%                     disp(beta)
%                     disp("lynx_d");
%                     disp(lynx_d);
%                     disp("rab_d");
%                     disp(rab_d);
%                     min_tt=tt;min_xi=xi;min_N=N;min_Ns=Ns;
%                 end
%             end
%         end
%     end
% end
% figure('Position', [0 0 1280 420]);
% subplot(1,3,1);
% plot(min_tt,min_xi);title('relative populations');
% xlabel('time (yrs)');legend('\xi_1', '\xi_2');
% subplot(1,3,2);plot(min_tt,min_N)
% title('Predator & Prey Populations'); xlabel('time (yrs)');
% ylabel('Number in Population x 10^3');legend('Predator (M)', 'Prey (N)', 'Location', 'North')
% subplot(1,3,3);plot(min_N(:, 1), min_N(:,2), 'linewidth', 2); hold on
% hold on; plot(min_Ns(1), min_Ns(2), "+");
% plot(lynx, rab, 'r^'); hold on;
% title('Phase Plane Plot'); xlabel('Predator Population x 10^3');ylabel('Prey Population x 10^3');hold off
% set(findobj(gcf, '-property', 'FontSize'), "fontsize", 16)                
% % plot(lynx, pred_rab, 'b^'); hold off;
% % plot(x1+Ns(1),x2+Ns(2), 'k^');hold on; plot(x1+Ns(1), -x2+Ns(2), 'k^'); hold off;
% % legend('NL model', 'Linearized Model')
% 
% function yp=lotka1(t,y,p)
%     yp=diag([-p(2,2)+p(2,1)*p(1,2)*y(2), p(1,1)-p(1,2)*y(1)])*y;
% end

% 10.7
% k=0.5;alpha=0.02;
% gamma=1;beta=0.6;
% kappa=0.025;
% theta1=[k alpha;gamma beta];
% theta2=[k alpha;gamma beta-kappa];
% Ns1=[k/alpha;beta/(alpha*gamma)];
% Ns2=[k/alpha;(beta-kappa)/alpha/gamma];
% N00=[23;32];
% trange=[0 30];
% ts=1/100:1/100:60;
% Mss=ones(1,60*100)*Ns1(1);
% Nss1=ones(1,30*100)*Ns1(2);
% Nss2=ones(1,30*100)*Ns2(2);
% Nss=cat(2,Nss1,Nss2);
% [tt1,N1]=ode45(@(tt,N) lotka1(tt, N, theta1), trange, N00);
% % xi1=[N1(:,1)-Ns1(1),N1(:,2)-Ns1(2)];
% N01=[N1(end,1);N1(end,2)];
% [tt2, N2]=ode45(@(tt,N) lotka1(tt, N, theta2), trange, N01);
% % xi2=[N2(:,1)-Ns2(1),N2(:,2)-Ns(2)];
% 
% % x11min=min(N1(:, 1)-Ns1(1));
% % x11max=max(N1(:, 1)-Ns1(1));x12max=max(N1(:,2)-Ns1(2));
% % dx1=(x11max-x11min)/10;x1=x11min:dx1:x11max;
% % a1=x12max^2/Ns1(2);
% % % x12=real(sqrt((a-x1.^2/gamma/Ns1(1))*Ns1(2)));
% % 
% % x21min=min(N2(:, 1)-Ns2(1));
% % x21max=max(N2(:, 1)-Ns2(1));x22max=max(N2(:,2)-Ns2(2));
% % dx2=(x21max-x21min)/10;x2=x21min:dx2:x21max;
% % a2=x22max^2/Ns2(2);
% % x22=real(sqrt((a2-x1.^2/gamma/Ns2(1))*Ns2(2)));
% 
% tt=cat(1, tt1, tt2+30);N=cat(1, N1, N2);
% 
% figure('Position', [0 0 840 420]);
% subplot(1,2,1);plot(tt,N);hold on;
% plot(ts, Mss, 'b-'); hold on;
% plot(ts, Nss, 'r-');hold on;
% title('Predator & Prey Populations'); xlabel('time (yrs)');
% ylabel('Number in Population x 10^3');legend('Predator (M)', 'Prey (N)', 'Location', 'North')
% subplot(1,2,2);plot(N1(:, 1), N1(:,2), "b-", 'linewidth', 2); hold on
% plot(Ns1(1), Ns1(2), "b+");
% plot(N2(:, 1), N2(:,2), "r-", 'linewidth', 2);
% plot(Ns2(1), Ns2(2), 'r+');
% % axis square;
% title('Phase Plane Plot'); xlabel('Predator Population x 10^3');ylabel('Prey Population x 10^3');hold off
% set(findobj(gcf, '-property', 'FontSize'), "fontsize", 16)   
% % legend('NL model', 'Linearized Model')
% 
% function yp=lotka1(t,y,p)
%     yp=diag([-p(2,2)+p(2,1)*p(1,2)*y(2), p(1,1)-p(1,2)*y(1)])*y;
% end

% 10.8
N0=[761;1;1];
alpha=0.0022;gamma=0.455;
% ks=[0 0.01 0.05 0.1 0.25 0.5 0.75 1];
ks=[0.03 0.3];
% figure('Position', [0 0 2560 1280])
figure('Position', [0 0 840 420])
i=1;
for k=ks
    theta=[alpha;gamma;k];
    trange=[0 100];
    [tt, N]=ode45(@(tt,N) SIRS(tt,N,theta),trange,N0);
    rs=N/sum(N0);
%     subplot(2, 4, i);
    subplot(1, 2, i)
    plot(tt, rs, 'linewidth', 2);
    title('SIRS Sub-Poplution');
    xlabel(sprintf('time (days) (k=%.2f)', k));ylabel('ratio (%)');
    set(gca, 'fontsize', 16);
    i=i+1;
    disp("Max I: "); disp(num2str(max(N(:, 2))));
    disp("stable I:");disp(num2str(N(end, 2)));
end
legend("S", "I", "R");
% 
% k=0.01;theta=[alpha;gamma;k];
% trange=[0 200];
% [tt, N]=ode45(@(tt,N) SIRS(tt,N,theta),trange,N0);
% rs=N/sum(N0);
% figure;
% plot(tt, rs, 'linewidth', 2);
% title('SIRS Sub-Poplution');
% xlabel(sprintf('time (days) (k=%.2f)', k));ylabel('ratio (%)');
% set(gca, 'fontsize', 16);
% legend("S", "I", "R");
% %     
function z=SIRS(t,y,p)
    z=([-p(1)*y(2),0,p(3);p(1)*y(2),-p(2),0;0,p(2),-p(3)])*y;
end

% alpha=0.0022;gamma=0.455;k=0.03;N=763;
% Ns=[gamma/alpha;k*(N-gamma/alpha)/(gamma+k);(N-gamma/alpha)/(1+k/gamma)];
% A=[-alpha*Ns(2) -alpha*Ns(1) k;alpha*Ns(2) alpha*Ns(1)-gamma 0;0 gamma -k];
% 
% a=k+alpha*Ns(2);
% b=alpha*Ns(2)*(k+gamma);
% lam1=(-a+sqrt(a^2-4*b))/2;
% lam2=(-a-sqrt(a^2-4*b))/2;
