Stats=1e6; %set Stats large...
switch_count=0; noswitch_count=0; %set 0 at the beginning
for n = 1:Stats
a = randperm(3); %Monty places two goats and the car at random
%a(1) -goat, a(2) -goat, a(3) - car
i= floor(3.*rand)+1; %you select the door
% SWITCH STRATEGY
if(i == a(1)) switch_count=switch_count+1; %a(2)-opened, switch to a(3), car!
elseif (i == a(2)) switch_count = switch_count + 1;%a(1) opened, switch to a(3), car!
else switch_count = switch_count + 0; %a(1)/a(2) opened, switch to a(2)/a(1), no car :-(
end
% NO SWITCH STRATEGY
if(i == a(1)) noswitch_count = noswitch_count + 0; %a(2)-opened, no car :-(
elseif (i==a(2)) noswitch_count = noswitch_count + 0; %a(1)-opened, no car :-(
else  noswitch_count = noswitch_count + 1; %a(1) or a(2)-opened, car!
end
end;
disp('probability to win a car if switched doors=');
disp(num2str(switch_count./Stats)); %# of cars with switching
disp('probability to win a car if did not switch doors=');
disp(num2str(noswitch_count./Stats)); %# of cars w/o switching