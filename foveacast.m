AL = 0.62171;     %���x��1��LL
%AL = 0.34537;
%A = 0.67234;     %���x��1��LH,HL
%A = 0.72709;     %���x��1��HH
a = 0.495;       %�p�ӂ��ꂽ�l
k = 0.466;       %�p�ӂ��ꂽ�l
gs = 1.501;      %�p�ӂ��ꂽ�l(LL)
%gs = 1;          %�p�ӂ��ꂽ�l(LH,HL)
%gs = 0.534;      %�p�ӂ��ꂽ�l(HH)
fs = 0.401;      %�p�ӂ��ꂽ�l
%x1 = 80;        %�C�ӂ̒l�i��f�l�j
%x2 = 0;        %�C�ӂ̒l�i��f�l�j
seen = zeros(512,512,'double');
v = 3;           %����
d = 1;           %���x��
%N = 285.35;         %��f�l
N = 512;
xf1 = 256;        %�����_
xf2 = 256;        %�����_
LL1 = xf1/2^d;    %LL��DWT-domain
LL2 = xf2/2^d;    %LL��DWT-domain
LH1 = (xf1+N)/2^d;   %LH��DWT-domain
LH2 = xf2/2^d;     %LH��DWT-domain
HL1 = xf1/2^d;     %HL��DWT-domain
HL2 = (xf2+N)/2^d;    %HL��DWT-domain
HH1 = (xf1+N)/2^d;     %HH��DWT-domain
HH2 = (xf2+N)/2^d;     %HH��DWT-domain
r = (N*v*pi)/180;          %�𑜓x
f = r*2^(-d);    %�\���𑜓xr�ƃE�F�[�u���b�g�������x���Ɋ֌W����
%Y = a*10^(k*(log10(r)-log10((2^d)*gs*fs))^2);
Y = a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2);
SNR = 10;         %�m�C�Y
sigma2 = 10.^(-SNR./10);
%rng(1);

B = 1;

%dx = sqrt((x1-(x1)^f)^2+(x2-(x2)^f)^2);
%%
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3842;

%Sw = 2*Y/A; %�Q�l�_������̊m�F�p

l = 1;
m = 1;
cut = 0;
for x1 = 1:1:256 %�_����B�̒l��1�`256�̉\��(LL)
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
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw1.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

%%
%level 2

d = 2;
AL = 0.34537;
gs = 1.501;      %�p�ӂ��ꂽ�l(LL)
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3818;

l = 1;
m = 1;
cut = 0;
for x1 = 1:1:128 %�_����B�̒l��1�`128�̉\��(LL)
    l = 1;
    for x2 = 1:1:128
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw1.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

%ALH = 0.67234;     %���x��1��LH,HL
ALH = 0.41317;
gs = 1;          %�p�ӂ��ꂽ�l(LH,HL)
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3326;
m = 1;

for x1 = 129:1:256 %�_����B�̒l��129�`256�̉\��(LH)
    l = 129;
    for x2 = 1:1:128
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end


for x1 = 1:1:128 %�_����B�̒l��1�`128�̉\��(HL)
    l = 1;
    for x2 = 129:1:256
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^B) * (Sf.^2.5);

seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

%AH = 0.72709;     %���x��1��HH
AH = 0.49428;
gs = 0.534;      %�p�ӂ��ꂽ�l(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2138;
m = 129;

for x1 = 129:1:256 %�_����B�̒l��129�`256�̉\��(HH)
    l = 129;
    for x2 = 129:1:256
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw3.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

%%
%level 3

d = 3;
AL = 0.18004;
gs = 1.501;      %�p�ӂ��ꂽ�l(LL)
Sw1 = AL/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2931;
l = 1;
m = 1;
cut = 0;
for x1 = 1:1:64 %�_����B�̒l��1�`64�̉\��(LL)
    l = 1;
    for x2 = 1:1:64
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw1.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

%ALH = 0.67234;     %���x��1��LH,HL
ALH = 0.22727;
gs = 1;          %�p�ӂ��ꂽ�l(LH,HL)
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.3019;
m = 1;

for x1 = 65:1:128 %�_����B�̒l��257�`512�̉\��(LH)
    l = 65;
    for x2 = 1:1:64
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-xf2/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end


for x1 = 1:1:64 %�_����B�̒l��1�`256�̉\��(HL)
    l = 1;
    for x2 = 65:1:128
    dx1 = (2^d)*sqrt(((x1-xf1/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^B) * (Sf.^2.5);

seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

%AH = 0.72709;     %���x��1��HH
AH = 0.28688;
gs = 0.534;      %�p�ӂ��ꂽ�l(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2442;
m = 65;

for x1 = 65:1:128 %�_����B�̒l��257�`512�̉\��(HH)
    l = 65;
    for x2 = 65:1:128
    dx1 = (2^d)*sqrt(((x1-(xf1+N)/2^d)^2)+((x2-(xf2+N)/2^d)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw3.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end


%%
d = 1;
ALH = 0.67234;     %���x��1��LH,HL
gs = 1;          %�p�ӂ��ꂽ�l(LH,HL)
%ALH = 0.41317;
Sw2 = ALH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.2700;
m = 1;

for x1 = 257:1:512 %�_����B�̒l��257�`512�̉\��(LH)
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
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end


for x1 = 1:1:256 %�_����B�̒l��1�`256�̉\��(HL)
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
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw2.^B) * (Sf.^2.5);

seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

AH = 0.72709;     %���x��1��HH
%AH = 0.49428;
gs = 0.534;      %�p�ӂ��ꂽ�l(HH)
Sw3 = AH/(a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2));
%0.1316;
m = 257;

for x1 = 257:1:512 %�_����B�̒l��257�`512�̉\��(HH)
    l = 257;
    for x2 = 257:1:512
    dx1 = (2^d)*sqrt(((x1-HH1)^2)+((x2-HH2)^2));
    %dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx1/(N*v));%�x
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:256) = min(fc,fd);
    fm = min(2.3*log(64)/(0.106*(2.3+atan(dx1/(N*v)))), (pi*N*v)/360);%���W�A��
if f <= fm
Sf = exp(-0.0461*f*atand(dx1/(N*v)));
%Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw3.^B) * (Sf.^2.5);
seen(m,l) = S;
l = l+1;
    end
    m = m+1;
end

plot(cut)
%stem(cut,'filled')
%gi = sqrt(P*sqrt(Sn)/sqrt(ramdan*symsum(sqrt(ramdan*Sn, n))));

%�����܂łŒ����_(seen)�̌v�Z�I��
%%

load('./foreman2_result/ydct_frame_qp_LL43_gop_1.mat');
load('./foreman2_result/ydct_frame_qp_LH43_gop_1.mat');
load('./foreman2_result/ydct_frame_qp_HL43_gop_1.mat');
load('./foreman2_result/ydct_frame_qp_HH43_gop_1.mat');

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
    gi = 0;
    
    gi = sqrt(length(S_total)) .* sqrt(sqrt(S_total./ramda_total)) .* sqrt(P/sum(sqrt(ramda_total.*S_total)));
    gi_exist = sqrt(length(S_total)) .* sqrt(sqrt(1./ramda_total)) .* sqrt(P/sum(sqrt(ramda_total)));

    M = sum(sum(gi.^2.*ramda_total))/length(S_total); %�m�F�p = 1 �ƂȂ�

    %X = dwt_total .* gi;
    X = dwt_total .* gi_exist;
    Ni = randn(size(X)) .* sqrt(sigma2);%���K���z��10dB�Ȃ�N(0,0.1)�̗���
    Z = X + Ni;
    %Z = dwt_total;
    %hat_c = ((ramda_total.*gi) ./ (ramda_total.*(gi).^2+sigma2)) .* Z;
    hat_c = ((ramda_total.*gi_exist) ./ (ramda_total.*(gi_exist).^2+sigma2)) .* Z;
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
    %Residual(:,:,frames) = idwt2D(hat_w,1,sf);
    %Orig(:,:,frames) = idwt2D(orig_hat_w,1,sf);
    Residual(:,:,frames)  = wavecdf97(hat_w,-3);
    Orig(:,:,frames) = wavecdf97(orig_w,-3);
end


Decoded = Residual(:,:,1);
MSE = (Orig(:,:,1) - Residual(:,:,1)).^2;