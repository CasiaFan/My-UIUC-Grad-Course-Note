% Stats=100000; r=0.1;
% r2=random('Exponential', 1/r, Stats,1); 
% disp([mean(r2),1./r]); disp([std(r2),1./r]);
% %
% step=0.1; [a,b]=hist(r2,0:step:max(r2));
% pdf_e=a./sum(a)./ step;
% figure;
% subplot(1,2,1); semilogy(b,pdf_e,'ko-'); hold on; 
% %%
% x=0:0.01:130; 
% for m=1:length(x);   
%     ccdf_e(m)=sum(r2 > x(m))./Stats; 
% end;
% subplot(1,2,2);  plot(x,ccdf_e,'ko-'); hold on; 
%%
% Stats=100000; r=0.1; k=9.75;
% r2=random('Gamma', k,??, Stats,1);
% disp([mean(r2),k./r]); 
% disp([std(r2),sqrt(k)./r]);
% step=0.1; [a,b]=hist(r2,0:step:max(r2));
% pdf_g=a./sum(a) ??./ or .*?? step;
% %%
% hold on; subplot(1,2,1); semilogy(b,pdf_g,'rd-'); hold on; 
% x=0:0.01:100; 
% for m=1:length(x);   
%     cdf_g(m)=sum(r2 ?? x(m))./Stats; 
% end;
% subplot(1,2,2); hold on; semilogy(x,cdf_g,'rd-'); hold on; 

% x=0:0.001:8;
% err=3.4e-6;
% 
% res_2 = 1-normcdf(6, 0, 1);
% 
% y=1-normcdf(x, 0, 1)-err;
% 
% find(abs(y)<1e-8)

% load("PINT_binding_energy.mat");
% eng=binding_energy(binding_energy>-23);
% eng=eng(eng<-10);
% dfittool()

% stats = 1000;
% r1 = normrnd(0, 2, [1, stats]);
% r2 = normrnd(0, 2, [1, stats]);
% figure;
% plot(r1, r2, 'k.');
% 
% figure;
% mix = 0.9;
% r1mix = mix.*r2 + (1-mix^2)^0.5.*r1;
% plot(r1mix, r2, 'k.'); hold on;
% 
% figure;
% mix2 = -0.5;
% r1mix2 = mix2.*r2 + (1-mix2^2)^0.5.*r1;
% plot(r1mix2, r2, 'k.'); hold off;
% 
% corr(r1, r2);
