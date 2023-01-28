% t=0:0.01:10;A0=1;u0=1;w0=2*pi*u0;phi=0;N=length(t);
% theta1=50;theta2=60;e0a=0.1;e0b=0.1;e0c=0.1;
% x=A0*cos(w0*t+phi);plot(t,x,'k');hold on;
% ea=e0a*randn(size(t));eb=e0b*randn(size(t));
% xa=x*cosd(theta1)+ea;xb=x*cosd(theta2)+eb;
% ec=e0c*randn(size(t));xc=ec;
% plot(t,xa, '.r');plot(t,xb,'.b');plot(t,xc,'.k');hold off;
% X=[(xa-mean(xa))' (xb-mean(xb))' (xc-mean(xc))'];
% Kx=X'*X/(N-1);
% figure;subplot(3,1,1);plot(xa,xb,'.k');title('x_A,x_B')
% subplot(3,1,2);plot(xb,xc,'.k');title('x_A, x_C')
% subplot(3,1,3);plot(xb,xc,'.k');title('x_B,x_C')
% [U, S]=eig(Kx);Y=U'*X';Ky=Y*Y'/(N-1);
% figure('Position', [0, 0, 320, 960]);subplot(3,1,1);plot(Y(1,:),Y(2,:),'.k');title('y_2, y_3')
% subplot(3,1,2);plot(Y(1,:), Y(3,:), '.k');title('y_1, y_3')
% subplot(3,1,3);plot(Y(2,:), Y(3,:), '.k');title('y_2, y_3')
% figure;plot(t, Y(3,:), 'ko');hold on; plot(t,x,'k')
% plot(t, Y(2,:), 'kx');plot(t,Y(1,:),'k.');hold off;

% t=-20:0.01:20;T0=10;c=2;h=1;q=zeros(size(t));
% for k = 1:500
%     q=q+sin(pi*k*c/T0)*cos(2*pi*k*t/T0)/(pi*k*c/T0);
% end
% g=(h*c/T0)*(1+2*q);plot(t,g);

% M0=1;M1=1;t1=1;u1=1;M2=0.5;t2=2;u2=2;M3=0.333;t3=3;u3=3;
% u=0:0.01:5;
% total = zeros(1,length(u));
% res1 = real(t1, M1, u1, u);
% res2 = real(t2, M2, u2, u);
% res3 = real(t3, M3, u3, u);
% total= total + res1 + res2 + res3;
% plot(u, total, 'ko-'); hold on;
% plot([5, 0], [0, 0], "--");
% set(gca, "FontSize", 16);
% xlabel("u", "Fontsize", 20);
% ylabel("$\mathcal{F}FID(t)$", "FontSize", 20, "Interpreter", "latex");
% 
% function res = real(t0, m0, u0, u)
%     o=2*pi*u;
%     o0=2*pi*u0;
%     a=1/t0;
%     res=m0*(a*(a+o0^2-o.^2) + 2*o*t0)./((a^2+o0^2-o.^2).^2+(2*o*t0).^2);
% end

% u=0:0.1:100;u0=60;sig=3/u0;
% gu=exp(-18*pi^2*u.^2/60^2);
% delta = zeros(1, length(u));
% delta(599:601)=10*0.25;
% plot(u, gu+delta, ".k-");
% set(gca, "FontSize", 16);
% xlabel("u", "Fontsize", 20);
% ylabel("$\mathcal{F}g(t)$", "FontSize", 20, "Interpreter", "latex");

% u=-0:0.01:10;sig=1;t0=2;a=0.2;u0=5;amp=1.5;
% fu = amp*exp(-(u-u0).^2/sig^2);
% gu = fu.*(1+a*cos(2*pi*u*t0));
% plot(u, fu, "Linewidth", 2); hold on;
% plot(u, gu, "Linewidth", 2);
% set(gca, "FontSize", 16);
% xlabel("u", "Fontsize", 20);
% ylabel("$\mathcal{G}(u)$ vs $\mathcal{F}(u)$", "FontSize", 20, "Interpreter", "latex");
% t=1:1:100;lambda1=2;lambda2=3;
% x=power(lambda1, t)./factorial(t)*exp(-lambda1);
% y=power(lambda2, t)./factorial(t)*exp(-lambda2);
% z=x+y;
% plot(t,x); hold on;
% plot(t,y); hold on;
% plot(t,z);
t=0:0.01:2;
y1=t;
y2=0.42*exp(t);
plot(t,y1); hold on;
plot(t,y2)

