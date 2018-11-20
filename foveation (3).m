AL = 0.62171;     %レベル1のLL
%A = 0.67234;     %レベル1のLH,HL
%A = 0.72709;     %レベル1のHH
%A = 0.34537;     %レベル2のLL
%A = 0.41317;     %レベル2のLH,HL
%A = 0.49428;     %レベル2のHH
%A = 0.28688;     %レベル3のHH
%A = 0.11792;     %レベル4のLH,HL
%A = 0.045943;    %レベル5のLL
%A = 0.030018;    %レベル6のLH,HL
a = 0.495;       %用意された値
k = 0.466;       %用意された値
gs = 1.501;      %用意された値(LL)
%gs = 1;          %用意された値(LH,HL)
%gs = 0.534;      %用意された値(HH)
fs = 0.401;      %用意された値
%x1 = 80;        %任意の値（画素値）
%x2 = 0;        %任意の値（画素値）
seen = zeros(512,512,'double');
v = 3;           %距離
d = 1;           %レベル
%N = 285.35;         %画素値
N = 512;
xf1 = 100;
xf2 = 100;
LL1 = xf1/2^d;    %LLのDWT-domain
LL2 = xf2/2^d;    %LLのDWT-domain
LH1 = xf1+N/2^d;   %LHのDWT-domain
LH2 = xf2/2^d;     %LHのDWT-domain
HL1 = xf1/2^d;     %HLのDWT-domain
HL2 = xf2+N/2^d;    %HLのDWT-domain
HH1 = xf1+N/2^d;     %HHのDWT-domain
HH2 = xf2+N/2^d;     %HHのDWT-domain
r = (N*v*pi)/180;          %解像度
f = r*2^(-d);    %表示解像度rとウェーブレット分解レベルに関係する
%Y = a*10^(k*(log10(r)-log10((2^d)*gs*fs))^2);
Y = a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2);

%dx = sqrt((x1-(x1)^f)^2+(x2-(x2)^f)^2);

Sw1 = AL/Y;
%Sw = 2*Y/A; %参考論文からの確認用

l = 1;
m = 1;
cut = 0;
for x1 = 1:1:256 %論文内Bの値が1～256の可能性(LL)
    l = 1;
    for x2 = 1:1:256
    dx1 = (2^d)*sqrt(((x1-LL1)^2)+((x2-LL2)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/0.106*(2.3+atan(dx1/N*v)), pi*N*v/360);
if f <= fm
Sf = exp(-0.0461*f*atan(dx1/N*v));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw1.^1) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

ALH = 0.67234;     %レベル1のLH,HL
Sw2 = ALH/Y;
m = 1;

for x1 = 257:1:512 %論文内Bの値が257～512の可能性(LH)
    l = 257;
    for x2 = 1:1:256
    dx1 = (2^d)*sqrt(((x1-LH1)^2)+((x2-LH2)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/0.106*(2.3+atan(dx1/N*v)), pi*N*v/360);
if f <= fm
Sf = exp(-0.0461*f*atan(dx1/N*v));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^1) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end


for x1 = 1:1:256 %論文内Bの値が1～256の可能性(HL)
    l = 1;
    for x2 = 257:1:512
    dx1 = (2^d)*sqrt(((x1-HL1)^2)+((x2-HL2)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/0.106*(2.3+atan(dx1/N*v)), pi*N*v/360);
if f <= fm
Sf = exp(-0.0461*f*atan(dx1/N*v));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^1) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.72709;     %レベル1のHH
Sw3 = AH/Y;
m = 257;

for x1 = 257:1:512 %論文内Bの値が257～512の可能性(HH)
    l = 257;
    for x2 = 257:1:512
    dx1 = (2^d)*sqrt(((x1-HH1)^2)+((x2-HH2)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/0.106*(2.3+atan(dx1/N*v)), pi*N*v/360);
if f <= fm
Sf = exp(-0.0461*f*atan(dx1/N*v));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw3.^1) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

plot(cut)
%stem(cut,'filled')
%gi = sqrt(P*sqrt(Sn)/sqrt(ramdan*symsum(sqrt(ramdan*Sn, n))));
%ここまでで注視点(seen)の計算終了

domain = randn(512,512);
n = 1;
for l = 1:1:256 
    for m = 1:1:256
ramda1(1,n) = domain(m,l)^2;
s1(1,n) = seen(m,l);
n = n+1;
    end
end

n = 1;
for l = 257:1:512 
    for m = 1:1:256
ramda2(1,n) = domain(m,l)^2;
s2(1,n) = seen(m,l);
n = n+1;
    end
end

n = 1;
for l = 1:1:256 
    for m = 257:1:512
ramda3(1,n) = domain(m,l)^2;
s3(1,n) = seen(m,l);
n = n+1;
    end
end

n = 1;
for l = 257:1:512 
    for m = 257:1:512
ramda4(1,n) = domain(m,l)^2;
s4(1,n) = seen(m,l);
n = n+1;
    end
end

S_total = [s1 s2 s3 s4];
ramda_total = [ramda1 ramda2 ramda3 ramda4];
P = 1;
gi = 0;
gi1 = 0;
gi2 = 0;
gi3 = 0;
gi4 = 0;
%ramda1 = mean(mean(ramda1));

gi = sqrt(length(S_total)) .* sqrt(sqrt(S_total./ramda_total)) .* sqrt(P/sum(sqrt(ramda_total.*S_total)));


M = sum(sum(gi.^2.*ramda_total))/length(S_total); %確認用 = 1 となる
