Stats=10000000;
%%
% r0 is a continuous random number between 0 and 1 
% it is >0.5 with probability 0.5 (heads)
% and is < 0.5 with probability 0.5 (tails)
% we need to convert it to r1, which is 0 or 1
r0=rand(Stats,1);
%%
% r1(t) is 0 if it was tails at time t and 1 if it was heads
% check help pages for function "floor"
r1=floor(r0.*2); 
%%
% n_+heards count the number of heads up to time t
n_heads(1)=r1(1);
for t=2:Stats; 
    n_heads(t)=n_heads(t-1)+r1(t);
end;
%%
tp=[1,10,100,1000,10000,100000,1000000,10000000]
%%
np=n_heads(tp); fp=np./tp
%%
figure; semilogx(tp,fp,'ko-');
%%
hold on; semilogx([1,10000000],[0.5,0.5],'r--');
%%
figure; loglog(tp,abs(fp-0.5),'ko');
%%
hold on; loglog(tp,0.5./sqrt(tp),'r--');

