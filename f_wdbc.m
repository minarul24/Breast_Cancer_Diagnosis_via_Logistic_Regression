%f_wdbc
function f = f_wdbc(w,D,mu)
P = size(D,2);
xp = D(1:31,:);
yp = D(32,:);
f = (1/P)*(sum(log(1+exp(-yp'.*xp'*w))))+(w'*w)*mu*0.5;
end