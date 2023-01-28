%% Generate data here (lab 5 assignment)
%
% t=0:0.1:10;a=0.5;x=1-exp(-a*t);rng('default');
% y1=x+0.2*randn(size(t));plot(t,y1,'.')%plot(t,x,t,y1,'.')
% y2=x+0.2/sqrt(10)*randn(size(t));figure;plot(t,y2,'.')
% save('lab5','t','y1','y2'); % save in lab5.mat
load("lab5.mat")

line_c = ["#808080" "#808000" "#FF0000" "#008000" "#800080" "#008080" "#00FFFF" "#FFFF00" "#FF00FF"];

% % mse plot
% m = 8;
% mse_v1 = zeros(1, m);
% mse_v2 = zeros(1, m);
% % polynomial function
% % plot(t, y1, 'o');
% for i = 0:m
%     p1 = polyfit(t, y1, i);
%     z1 = polyval(p1, t);
% %     hold on; plot(t, z1, 'b', 'linewidth', 2, 'color', line_c(i+1)) 
%     mse_v1(i+1) = meanse(y1, z1);
% end
% 
% % figure; plot(t, y2, 'o');
% for j = 0:m
%     p2 = polyfit(t, y2, j);
%     z2 = polyval(p2, t);
% %     hold on; plot(t, z2, 'b', 'linewidth', 2, 'color', line_c(j+1))
%     mse_v2(j+1) = meanse(y2, z2);
% end 
% 
% mse_v = cat(1, mse_v1, mse_v2);
% figure('Position', [0 0 1000 500])
% hist = bar(mse_v', 'grouped');
% legend([hist(1) hist(2)], {'$y_1$' '$y_2$'}, 'Interpreter', "latex");
% ht = [];
% hist_x1 = hist(1).XData+hist(1).XOffset;
% hist_y1 = hist(1).YData;
% hist_x2 = hist(2).XData+hist(2).XOffset;
% hist_y2 = hist(2).YData; 
% for i = 1:length(hist)
%     ht = [ht text(hist(i).XData+hist(i).XOffset, hist(i).YData, num2str(hist(i).YData.', '%.4f'), ...
%          'VerticalAlignment','bottom','horizontalalign','center')];
% end
% set(ht, 'FontSize', 14)
% xlabel('m', 'FontSize', 18);
% xticklabels(0:8);
% ylabel('MSE', 'FontSize', 18);
% set(gca, 'FontSize', 16)
% hold on; p1=plot(hist_x1, hist_y1, 'ko-', 'color', "#0072BD", "linewidth", 2);
% hold on; p2=plot(hist_x2, hist_y2, 'ko-', 'color', "#D95319", "linewidth", 2);
% set(get(get(p1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
% set(get(get(p2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')

% plot best m in fitting with polynomial function
best_m = input('select polynomial degree value m (0 to 8): ');
P_1b = polyfit(t, y1, best_m);
z_1b = polyval(P_1b, t);
mse_p1 = meanse(y1, z_1b)
% figure('Position', [0 0 480 480])
figure('Position', [0 0 1280 960]);
sub_fig_s = 0.4

% subaxis(2,2,1, 'hs', 0.03, 'vs', 0.01, 'Padding', 0, 'Margin', 0);
subplot(2, 2, 1, 'Position', [0.05 0.56 sub_fig_s sub_fig_s]);
l1 = plot(t, y1, "+", 'color', "#0072BD", 'linewidth', 2);
hold on; l2 = plot(t, z_1b, '+-', 'linewidth', 2, 'color', "#D95319");
xlabel("$t$", "Interpreter", "latex");
ylabel("\textbf{output:} $\hat y_1$ vs $z(poly)$", "Interpreter", "latex");
set(gca, 'FontSize', 16)
text(0.25, 1.2, "(A)", 'fontsize', 20)
% title("(A)", 'FontSize', 20)

P_2b = polyfit(t, y2, best_m);
z_2b = polyval(P_2b, t);
mse_p2 = meanse(y2, z_2b)
% figure('Position', [0 0 480 480])
subplot(2, 2, 2, 'Position', [0.55 0.56 sub_fig_s sub_fig_s]);
plot(t, y2, "+", 'color', "#0072BD", 'linewidth', 2);
hold on; plot(t, z_2b, '+-', 'linewidth', 2, 'color', "#D95319")
xlabel("$t$", "Interpreter", "latex");
ylabel("\textbf{output:} $\hat y_2$ vs $z(poly)$", "Interpreter", "latex");
set(gca, 'FontSize', 16)
text(0.25, 0.97, "(B)", 'fontsize', 20)
% title("(B)", 'FontSize', 20)


% log fit
yy1 = log(1 - y1); 
p_log_1 = polyfit(t, yy1, 1);
zz1 = 1 - exp(real(polyval(p_log_1, t)));

mse_log_1 = meanse(y1, zz1)
subplot(2, 2, 3, 'Position', [0.05 0.1 sub_fig_s sub_fig_s]);
l3 = plot(t, y1, "+", 'color', "#0072BD", 'linewidth', 2);
hold on; l4 = plot(t, zz1, '+-', 'linewidth', 2, 'color', "#EDB120");
xlabel("$t$", "Interpreter", "latex");
ylabel("\textbf{output:} $\hat y_1$ vs $z(log)$", "Interpreter", "latex");
set(gca, 'FontSize', 16)
text(0.25, 1.2, "(C)", 'fontsize', 20)
% title("(C)", 'FontSize', 20)
%
yy2 = log(1 - y2); 
p_log_2 = polyfit(t, yy2, 1);
zz2 = 1 - exp(real(polyval(p_log_2, t)));

mse_log_2 = meanse(y2, zz2)
subplot(2, 2, 4, 'Position', [0.55 0.1 sub_fig_s sub_fig_s]);
plot(t, y2, "+", 'color', "#0072BD", 'linewidth', 2);
hold on; plot(t, zz2, '+-', 'linewidth', 2, 'color', "#EDB120")
xlabel("$t$", "Interpreter", "latex");
ylabel("\textbf{output:} $\hat y_2$ vs $z(log)$", "Interpreter", "latex");
set(gca, 'FontSize', 16)
text(0.25, 0.97, "(D)", 'fontsize', 20)
% title("(D)", 'FontSize', 20)
% pos = get(gca, 'Position')


lg = legend([l1, l2, l4], {'$\hat y$', '$z(poly)$', '$z(log)$'}, ...
    'Interpreter', 'latex', 'Orientation', 'horizontal');
set(lg, 'Box', 'off')
set(lg, 'Position', [0.4 0.0 0.2, 0.05])

corrcoef(t, y1)
corrcoef(t, y2)
function mse = meanse(input1, input2)
    num = length(input1);
    mse = sum(power(input1-input2, 2), 'all') / num;
end
