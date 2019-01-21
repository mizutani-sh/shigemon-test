
a = 0.495;       %用意された値
k = 0.466;       %用意された値
gs = 1.501;      %用意された値(LL)
%gs = 1;          %用意された値(LH,HL)
%gs = 0.534;      %用意された値(HH)
fs = 0.401;      %用意された値
%x1 = 80;        %任意の値（画素値）
%x2 = 0;        %任意の値（画素値）
seen = zeros(512,512,'double');
v = 3;
level = 5;
d = 1;           %レベル
N = 512;         %画素値
xf1 = 256;        %注視点
xf2 = 256;        %注視点
LL1 = xf1/2^d;    %LLのDWT-domain
LL2 = xf2/2^d;    %LLのDWT-domain
LH1 = (xf1+N)/2^d;   %LHのDWT-domain
LH2 = xf2/2^d;     %LHのDWT-domain
HL1 = xf1/2^d;     %HLのDWT-domain
HL2 = (xf2+N)/2^d;    %HLのDWT-domain
HH1 = (xf1+N)/2^d;     %HHのDWT-domain
HH2 = (xf2+N)/2^d;     %HHのDWT-domain
r = (N*v*pi)/180;          %解像度
f = r*2^(-d);    %表示解像度rとウェーブレット分解レベルに関係する
%Sw = 2*Y/A; %参考論文からの確認用
Y = a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2);
SNR = 10;
sigma2 = 10.^(-SNR./10);

B1 = 0.5;
B2 = 2.5;

%dx = sqrt((x1-(x1)^f)^2+(x2-(x2)^f)^2);
%%
AL = 0.62171;     %レベル1のLL
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3842;

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
    fm = min((2.3*log(64))/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atan(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw1.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

ALH = 0.67234;     %レベル1のLH,HL
gs = 1;          %用意された値(LH,HL)
%ALH = 0.41317;
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2700;
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
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
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
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);

seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.72709;     %レベル1のHH
%AH = 0.49428;
gs = 0.534;      %用意された値(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.1316;
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
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw3.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

%%
%level 2

if level >= 2
d = 2;
AL = 0.34537;
gs = 1.501;      %用意された値(LL)
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3818;

m = 1;
for x1 = 1:1:128 %論文内Bの値が1～128の可能性(LL)
    l = 1;
    for x2 = 1:1:128
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw1.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

ALH = 0.41317;    %レベル2のLH,HL
gs = 1;          %用意された値(LH,HL)
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3326;
m = 1;

for x1 = 129:1:256 %論文内Bの値が129～256の可能性(LH)
    l = 129;
    for x2 = 1:1:128
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

for x1 = 1:1:128 %論文内Bの値が1～128の可能性(HL)
    l = 1;
    for x2 = 129:1:256
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.49428;    %レベル2のHH
gs = 0.534;      %用意された値(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2138;
m = 129;

for x1 = 129:1:256 %論文内Bの値が129～256の可能性(HH)
    l = 129;
    for x2 = 129:1:256
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw3.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
end

%%
%level 3

if level >= 3
d = 3;
AL = 0.18004;    %レベル3のLL
gs = 1.501;      %用意された値(LL)
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2931;

m = 1;
for x1 = 1:1:64 %論文内Bの値が1～64の可能性(LL)
    l = 1;
    for x2 = 1:1:64
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw1.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

ALH = 0.22727;    %レベル3のLH,HL
gs = 1;          %用意された値(LH,HL)
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3019;
m = 1;

for x1 = 65:1:128 %論文内Bの値が65～128の可能性(LH)
    l = 65;
    for x2 = 1:1:64
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

for x1 = 1:1:64 %論文内Bの値が1～64の可能性(HL)
    l = 1;
    for x2 = 65:1:128
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);

seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.28688;    %レベル3のHH
gs = 0.534;      %用意された値(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2442;
m = 65;

for x1 = 65:1:128 %論文内Bの値が65～128の可能性(HH)
    l = 65;
    for x2 = 65:1:128
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw3.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
end


%%
%level 4

if level >= 4
d = 4;
AL = 0.091401;    %レベル4のLL
gs = 1.501;      %用意された値(LL)
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3818;

m = 1;
for x1 = 1:1:32 %論文内Bの値が1～32の可能性(LL)
    l = 1;
    for x2 = 1:1:32
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw1.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
   
ALH = 0.11792;    %レベル2のLH,HL
gs = 1;          %用意された値(LH,HL)
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3326;
m = 1;

for x1 = 33:1:64 %論文内Bの値が129～256の可能性(LH)
    l = 33;
    for x2 = 1:1:32
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

for x1 = 1:1:32 %論文内Bの値が1～32の可能性(HL)
    l = 1;
    for x2 = 33:1:64
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.15214;    %レベル4のHH
gs = 0.534;      %用意された値(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2138;
m = 33;

for x1 = 33:1:64 %論文内Bの値が129～256の可能性(HH)
    l = 33;
    for x2 = 33:1:64
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw3.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
end


%%
%level 5

if level >= 5
d = 5;
AL = 0.045943;    %レベル5のLL
gs = 1.501;      %用意された値(LL)
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3818;

m = 1;
for x1 = 1:1:16 %論文内Bの値が1～16の可能性(LL)
    l = 1;
    for x2 = 1:1:16
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw1.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
   
ALH = 0.059758;    %レベル5のLH,HL
gs = 1;          %用意された値(LH,HL)
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3326;
m = 1;

for x1 = 17:1:32 %論文内Bの値が129～256の可能性(LH)
    l = 17;
    for x2 = 1:1:16
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

for x1 = 1:1:16 %論文内Bの値が1～16の可能性(HL)
    l = 1;
    for x2 = 17:1:32
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.077727;    %レベル5のHH
gs = 0.534;      %用意された値(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2138;
m = 17;

for x1 = 17:1:32 %論文内Bの値が17～32の可能性(HH)
    l = 17;
    for x2 = 17:1:32
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw3.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
end


%%
%level 6

if level >= 6
d = 6;
AL = 0.023013;    %レベル6のLL
gs = 1.501;      %用意された値(LL)
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3818;

m = 1;
for x1 = 1:1:8 %論文内Bの値が1～8の可能性(LL)
    l = 1;
    for x2 = 1:1:8
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw1.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
   
ALH = 0.030018;    %レベル6のLH,HL
gs = 1;          %用意された値(LH,HL)
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3326;
m = 1;

for x1 = 9:1:16 %論文内Bの値が9～16の可能性(LH)
    l = 9;
    for x2 = 1:1:8
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

for x1 = 1:1:8 %論文内Bの値が1～32の可能性(HL)
    l = 1;
    for x2 = 9:1:16
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw2.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.039156;    %レベル6のHH
gs = 0.534;      %用意された値(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2138;
m = 9;

for x1 = 9:1:16 %論文内Bの値が9～16の可能性(HH)
    l = 9;
    for x2 = 9:1:16
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
else
Sf = 0;
end

S = (Sw3.^B1) * (Sf.^B2);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end
end

%%
plot(cut)
%stem(cut,'filled')

%ここまでで注視点(seen)の計算終了
%%

LL_load = [strcat('.\foreman',num2str(level),'_result\ydct_frame_qp_LL50_gop_1.mat')];
LH_load = [strcat('.\foreman',num2str(level),'_result\ydct_frame_qp_LH50_gop_1.mat')];
HL_load = [strcat('.\foreman',num2str(level),'_result\ydct_frame_qp_HL50_gop_1.mat')];
HH_load = [strcat('.\foreman',num2str(level),'_result\ydct_frame_qp_HH50_gop_1.mat')];
load(LL_load);
load(LH_load);
load(HL_load);
load(HH_load);
%load('./foreman1_result/ydct_frame_qp_LL50_gop_1.mat');
%load('./foreman1_result/ydct_frame_qp_LH50_gop_1.mat');
%load('./foreman1_result/ydct_frame_qp_HL50_gop_1.mat');
%load('./foreman1_result/ydct_frame_qp_HH50_gop_1.mat');

for frames = 1:8
    LL = Y_LL(:,:,frames);
    LH = Y_LH(:,:,frames);
    HL = Y_HL(:,:,frames);
    HH = Y_HH(:,:,frames);
    
    orig_w(1:256,1:256) = LL;
    orig_w(1:256,257:512) = LH;
    orig_w(257:512,1:256) = HL;
    orig_w(257:512,257:512) = HH;
    
    dwt_total = [LL(:).' LH(:).' HL(:).' HH(:).'];

    n = 1;
    for l = 1:1:256 
        for m = 1:1:256
    s1(1,n) = seen(l,m);
    n = n+1;
        end
    end
    ramda_LL = LL.^2;
    ramda_LL = reshape(ramda_LL,1,[]);
    
    n = 1;
    for l = 1:1:256 
        for m = 257:1:512
    s2(1,n) = seen(l,m);
    n = n+1;
        end
    end
    ramda_LH = LH.^2;
    ramda_LH = reshape(ramda_LH,1,[]);
    
    n = 1;
    for l = 257:1:512 
        for m = 1:1:256
    s3(1,n) = seen(l,m);
    n = n+1;
        end
    end
    ramda_HL = HL.^2;
    ramda_HL = reshape(ramda_HL,1,[]);
    n = 1;
    for l = 257:1:512 
        for m = 257:1:512
    s4(1,n) = seen(l,m);
    n = n+1;
        end
    end
    ramda_HH = HH.^2;
    ramda_HH = reshape(ramda_HH,1,[]);
    
    S_total = [s1 s2 s3 s4];
    ramda_total = [ramda_LL ramda_LH ramda_HL ramda_HH];
    P = 1;
    
    gi = sqrt(length(S_total)) .* sqrt(sqrt(S_total./ramda_total)) .* sqrt(P/sum(sqrt(ramda_total.*S_total)));
    gi_exist = sqrt(length(S_total)) .* sqrt(sqrt(1./ramda_total)) .* sqrt(P/sum(sqrt(ramda_total)));

    M = sum(sum(gi.^2.*ramda_total))/length(S_total); %確認用 = 1 となる

    X = dwt_total .* gi;
    %X = dwt_total .* gi_exist;
    Ni = randn(size(X)) .* sqrt(sigma2);%正規分布の10dBならN(0,0.1)の乱数
    Z = X + Ni;
    %Z = dwt_total;
    hat_c = ((ramda_total.*gi) ./ (ramda_total.*(gi).^2+sigma2)) .* Z;
    %hat_c = ((ramda_total.*gi_exist) ./ (ramda_total.*(gi_exist).^2+sigma2)) .* Z;
    %hat_c = Z ./ gi;
    
    n = 1;
    for l = 1:1:256 
        for m = 1:1:256
    hat_LL(m,l) = hat_c(1,n);
    gi_LL(m,l) = gi(1,n);
    n = n+1;
        end
    end
    
    for l = 1:1:256 
        for m = 1:1:256
    hat_LH(m,l) = hat_c(1,n);
    gi_LH(m,l) = gi(1,n);
    n = n+1;
        end
    end
    
    for l = 1:1:256 
        for m = 1:1:256
    hat_HL(m,l) = hat_c(1,n);
    gi_HL(m,l) = gi(1,n);
    n = n+1;
        end
    end
    
    for l = 1:1:256 
        for m = 1:1:256
    hat_HH(m,l) = hat_c(1,n);
    gi_HH(m,l) = gi(1,n);
    n = n+1;
        end
    end
    
    %hat_w{2} = hat_LL;
    %hat_w{1}{1} = hat_LH;
    %hat_w{1}{2} = hat_HL;
    %hat_w{1}{3} = hat_HH;
    hat_w(1:256,1:256) = hat_LL;
    hat_w(1:256,257:512) = hat_LH;
    hat_w(257:512,1:256) = hat_HL;
    hat_w(257:512,257:512) = hat_HH;
    
    gi_w(1:256,1:256) = gi_LL;
    gi_w(1:256,257:512) = gi_LH;
    gi_w(257:512,1:256) = gi_HL;
    gi_w(257:512,257:512) = gi_HH;
    
    MSE = (hat_w - orig_w).^2;
    FWD = sum(sum(MSE .* seen)) ./ sum(sum(seen));
    FSNR = 10*log10(255^2./FWD);
    %hat = hat_LL + hat_LH + hat_HL + hat_HH;
    %hat_w = imresize(double(hat), [512,512]);
    %orig_hat_w{2} = LL;
    %orig_hat_w{1}{1} = LH;
    %orig_hat_w{1}{2} = HL;
    %orig_hat_w{1}{3} = HH;
    Residual(:,:,frames)  = wavecdf97(hat_w,-level);
    Orig(:,:,frames) = wavecdf97(orig_w,-level);
end

Digital = load('foreman_result/ycameras_qp_50_gop_1.mat');
Decoded = Digital.Y_Camera(:,:,1) + Residual(:,:,1);
Original = Digital.Y_Camera(:,:,1) + Orig(:,:,1);
MSE = (Orig(:,:,1) - Residual(:,:,1)).^2;
load('foreman_result/ysize_qp_50_gop_1.mat');

Residual_size = length(dwt_total) / 2; %ニアアナログのデータ量
%%
%以下デジタルのデータ量
TEST = load('foreman_result/bitstream_qp_50_gop_1.mat');

for x = 1:238224
    %{
        if TEST.bitstream(x) == '0'
            TEST.bitstream(x) = -1;
            disp(TEST.bitstream(x));
            
        end
    %}
    if TEST.bitstream(x) == '0'
        Testnum(x) = -1;
    else
        Testnum(x) = 1;
    end
end
plot(Testnum)