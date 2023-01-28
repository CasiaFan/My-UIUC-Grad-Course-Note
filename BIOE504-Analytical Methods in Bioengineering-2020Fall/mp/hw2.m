%% 4.2 
% syms tau;
% tau = 5;
% t=-10:0.01:10;
% length(t)
% s=1;
% gt = exp(-power(t, 2)/(2*s^2));
% % plot(t, gt)
% psi=sqrt(pi*s^2)*exp(-power(t-tau, 2)/(4*s^2));
% % hold on; plot(t, psi)
% 
% % y12 = xcorr(gt, psi) * 0.01;
% % yy12 = y12(1001:3001);
% % y21 = xcorr(psi, gt) * 0.01;
% % yy21 = y21(1001:3001);
% % [cst, lags] = xcorr(gt, gt);
% % plot(lags*0.01, cst*0.01);
% % figure;plot(t, yy12);plot(t,yy21)
% figure("Position", [0 0 640 640]);
% subplot(2, 1, 1);
% plot(t, gt, 'linewidth', 2);
% hold on; plot(t, psi, 'linewidth', 2);
% xlabel("t", 'FontSize', 18);
% set(gca, 'FontSize', 16);
% hold on; plot([tau, tau], [0, s*sqrt(pi)], "--");
% hold on; plot([-10, tau], [s*sqrt(pi), s*sqrt(pi)], "--");
% text(tau, 0.1, "$\tau_0$", "Interpreter", "latex", "FontSize", 20)
% text(-9.8, s*sqrt(pi), "$\sigma\sqrt{\pi}$", "Interpreter", "latex", 'FontSize', 20);
% 
% gt2 = exp(-power(t+tau, 2)/(2*s^2));
% % plot(t, gt)
% psi2=sqrt(pi*s^2)*exp(-power(t, 2)/(4*s^2));
% subplot(2, 1, 2);
% plot(t, gt2, 'linewidth', 2);
% hold on; plot(t, psi2, 'linewidth', 2);
% xlabel("$\tau$", "Interpreter", "latex", "Fontsize", 18);
% set(gca, "Fontsize", 16);
% hold on;plot([-tau, -tau], [0, 1], "--");
% hold on; plot([-10, -tau], [1, 1], "--");
% text(-tau, 0.1, "$\tau_0$", "Interpreter", "latex", "FontSize", 20)
% text(-9.8, 1, "1",'FontSize', 20);


% N=101;f=zeros(N);x0=floor(2*N/3);y0=x0;
% b=1;c=2;d=0.8;e=1.8;
% for j=1:N
%     for i=1:N
%         if(j<floor(N/3)+sqrt(floor(N/10)^2-(i-floor(N/3))^2)...
%                 && j > floor(N/3)-sqrt(floor(N/10)^2-(i-floor(N/3))^2))
%             f(j,i)=1;
%         end
%         if (j<floor(N/3)+sqrt(floor(N/12)^2-(i-floor(N/3))^2)...
%                 && j > floor(N/3)-sqrt(floor(N/12)^2-(i-floor(N/3))^2))
%             f(j,i)=0;
%         end
%         if(j<y0+sqrt((c*10)^2-(c*(i-x0)/b)^2)...
%                 && j >y0-sqrt((c*10)^2-(c*(i-x0)/b)^2))
%             f(j,i) =1;
%         end
%         if(j<y0+sqrt((e*10)^2-(e*(i-x0)/d)^2)...
%                 && j >y0-sqrt((e*10)^2-(e*(i-x0)/d)^2))
%             f(j,i) =0;
%         end
%     end
% end
% g = [f f'; f' f];
% rng('default');
% sig=poissrnd(10, 202, 202);
% nois=poissrnd(20, 202, 202);
% gg=g.* sig+nois;
% 
% snr = var(gg) / var(nois) - 1;
% figure('Position', [0, 0, 840, 360]);
% subplot(1, 2, 1);
% imagesc(gg);colormap gray
% set(gca, 'xtick', [], 'ytick', []);
% % ggg = imbinarize(gg);
% % res = imfill(ggg, 'holes');
% % figure;imagesc(res)
% 
% out = zeros(size(gg));
% s=1;L=5;
% % m = max(ceil(3*s),(L-1)/2);
% m = 4;
% [x,y] = meshgrid(-m:m,-m:m); % non-rotated coordinate system, contains (0,0)
% theta=0:30:150;
% 
% for t=theta
%     % angle in radian
%     u = cos(t)*x - sin(t)*y;     % rotated coordinate system
%     v = sin(t)*x + cos(t)*y;     % rotated coordinate system
% %     N = (abs(u) <= 3*s) & (abs(v) <= L/2);   % domain
%     N = (abs(u)<=m) & (abs(v)<=m);
%     k = exp(-u.^2/(2*s.^2));     % kernel
%     k = k - mean(k(N));
%     k(~N) = 0;                   % set kernel outside of domain to 0
%     res = conv2(gg, k, 'same');
%     out = max(out, res);
% end
% % me = mean(out);
% % out(out<me) = 0;
% % figure;
% subplot(1, 2, 2);
% imagesc(out);colormap gray
% set(gca, 'xtick', [], 'ytick', [])
% var(out)/var(nois) - 1

% t = 0:0.01:9.99;
% n = length(t);
% g1 = zeros(n, 1);
% g2 = zeros(n, 1);
% g1(250:350) = 1;
% g2(200:400) = 2;
% phi1 = xcorr(g1, g2);
% phi2 = conv(g1, g2);
% plot(0:0.01:19.98, phi1)
% hold on; plot(0:0.01:19.98, phi2)


t=0:0.01:9.99;
fn=cos(2*pi.*t);
N = length(t);
h1 = zeros(N, 1);
h1(1) = 100;
H(1, :) = h1;
for j=2:N
    h = zeros(N, 1);
    h(1:j) = 1/(j*0.01);
    H(j, :) = h;
end
g = H*fn';
plot(t, g, "Linewidth", 2);
xlabel("t");
set(gca, "FontSize", 16);
hold on; plot([0,10], [0, 0], "--", "Linewidth", 1.5);
text(5, 10, "g(t)", "fontsize", 20)
%     H(j, 1:j-1) = h(N-j+2:N);
%     H(j,j:end) = h(1:N-j+1);
% end
% 
% g = H*fn';
% g = conv(fn, h1, 'same');
% plot(t, g)

% for i=1:length(startp)
%     p = i:0.01:i+w;
%     s(i) = sum(cos(2*pi.*p));
% end
% s
% figure;plot(startp, s)
            