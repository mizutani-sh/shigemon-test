%A = 0.62171;     %���x��1��LL
%A = 0.67234;     %���x��1��LH,HL
%A = 0.72709;     %���x��1��HH
%A = 0.34537;     %���x��2��LL
%A = 0.41317;     %���x��2��LH,HL
%A = 0.49428;     %���x��2��HH
%A = 0.28688;     %���x��3��HH
%A = 0.11792;     %���x��4��LH,HL
A = 0.045943;    %���x��5��LL
%A = 0.030018;    %���x��6��LH,HL
a = 0.495;       %�p�ӂ��ꂽ�l
k = 0.466;       %�p�ӂ��ꂽ�l
gs = 1.501;      %�p�ӂ��ꂽ�l(LL)
%gs = 1;          %�p�ӂ��ꂽ�l(LH,HL)
%gs = 0.534;      %�p�ӂ��ꂽ�l(HH)
fs = 0.401;      %�p�ӂ��ꂽ�l
%x1 = 80;        %�C�ӂ̒l�i��f�l�j
x2 = 0;        %�C�ӂ̒l�i��f�l�j
v = 1;           %����
d = 5;           %���x��
%N = 285.35;         %��f�l
N = 512;
xf1 = 0;
xf2 = 0;
r = (N*v*pi)/180;          %�𑜓x
f = r*2^(-d);    %�\���𑜓xr�ƃE�F�[�u���b�g�������x���Ɋ֌W����
%Y = a*10^(k*(log10(r)-log10((2^d)*gs*fs))^2);
Y = a*10^(k*(log10(r*2^(-d))-log10(gs*fs))^2);

%dx = sqrt((x1-(x1)^f)^2+(x2-(x2)^f)^2);

Sw = A/Y;
%Sw = 2*Y/A; %�Q�l�_������̊m�F�p

l = 1;
for x1 = -256:1:256
    dx = sqrt(((x1-xf1)^2)+((x2-xf2)^2));
    e = atand(dx/(N*v));
    fc = (2.3*log(64))/(0.106*(e+2.3));
    fd = (pi*N*v)/360;
    cut(l:1:512) = min(fc,fd);
    l = l+1;
end
plot(cut)
%stem(cut,'filled')

fm = min(2.3*log(64)/0.106*(2.3+atan(dx/N*v)), pi*N*v/360);
if f <= fm
Sf = exp(-0.0461*f*atan(dx/N*v));
else
Sf = 0;
end

S = (Sw^1) * (Sf^2.5);

