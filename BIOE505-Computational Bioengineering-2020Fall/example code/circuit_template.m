Stats=1e6;
count= 0;
for i = 1:Stats
e1 = rand < 0.9; 
e2 = rand < 0.5; 
e3 = rand < 0.3;
e4 = rand < 0.1; 
e5 = rand < 0.4; 
e6 = rand < 0.5;
e7 = rand < 0.8;
s1 = and(e2,e3); 
s2 = or(e5,e6); 
s3 = and(e4,s2);
s4 = or(s1,s3); % 
s5 = e1 * s4 * e7; % 
count = count + s5;
end
p_circuit_works = count./Stats
%our calculation: P(circuit_works)=
%= 0.9.*(1-(1-0.5.*0.3).*(1-0.1.*(1-0.6.*0.5))).*0.8=
%=0.15084