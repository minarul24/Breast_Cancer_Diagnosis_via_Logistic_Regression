%function g_wdbc
function g = g_wdbc(w,D,mu)
P = size(D,2);
xp = D(1:31,:);
yp = D(32,:);
g=mu*w - sum(yp.*xp./(1+exp(yp'.*xp'*w))',2)/P;
end