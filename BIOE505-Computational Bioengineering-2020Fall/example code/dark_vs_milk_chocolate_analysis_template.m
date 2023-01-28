dark=[118.8 122.6 115.6 113.6 119.5 115.9 115.8 115.1 116.9 115.4 115.6 107.9];
milk=[102.1 105.8 99.6 102.7 98.8 100.9 102.8 98.7 94.7 97.8 99.7 98.6];
%%
x_dark=mean(dark) % sample mean dark chocolate
x_milk=mean(milk) % sample mean milk chocolate
%%
s_dark=std(dark) % sample std dark chocolate
s_milk=std(milk) % sample std milk chocolate
%%
n=12 % % sample size of both dark and milk
std_xdiff=sqrt(s_dark.^2 / n + s_milk.^2 / n) % std diff x
%%
z_stat=(x_dark - x_milk)./std_xdiff % t statistic 
P_value_z=1-normcdf(z_stat) %P-value that null is true
%%
dof=(n-1)+(n-1)  % # of degrees of freedom
P_value_t=tcdf(z_stat,dof,'upper') % P-value that null is true