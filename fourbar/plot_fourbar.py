import numpy as np
import matplotlib.pyplot as plt
import fourbar
import opt

pi = np.pi

ang = pi/3
l = [0.04,0.05,0.03,0.05,0.02]
kl = [0.1,0.1,0.1,0.1]
c = 1
dir = 1
gnd = 1

# ang = 1.3101957669671627
# l = [0.044267814832539915, 0.07437331623120089, 0.02988583253163232, 0.057249319999850376, 0.09412078333413701]
# kl = [0.05003576286243111, 0.3751956948522452, 0.2031977021784267, 0.49934374150383587]

# ang = 1.4383046660607426
# l = [0.062498115502122586, 0.09990103082818508, 0.05148357591437648, 0.09566723728887822, 0.09997223739765669]
# kl = [0.050007403769051606, 0.49801886574569554, 0.07376108406977244, 0.4992122584999922]

# ang = 1.6173432002792851
# l = [0.05431203641508329, 0.13434706427557624, 0.07252564975979359, 0.15584129852063744, 0.19999832399917888]
# kl = [0.05169821946130573, 0.453429189290968, 0.1664526118347955, 0.4984758119818369]

ang = 1.1898455290711798
l = [0.04716641817535354, 0.059993342087694745, 0.04045954425037288, 0.0598756548091391, 0.029591706587247624]
kl = [0.05008557965864202, 0.05060875118713612, 0.33129205651666127, 0.057191919007062525]
c = -0.6967929304224048
dir = 0.6994341539726643
gnd = -0.5129592154126137

ang -0.8924136912523644 l [0.06503067 0.09954018 0.07194137 0.07716939 0.07060763] k [0.05397111 0.15743529 0.45504484 0.20169808] c 0.9846970768851233 dir -0.13317981943955803 gnd -0.06263824845804433 convergence 0.04445776262535179
ang -0.76823249668265 l [0.07176029 0.09976059 0.06706698 0.07371429 0.07934666] k [0.05150974 0.15180582 0.49853449 0.13550389] c 0.07170923804003815 dir -0.3253483753006987 gnd -0.8464543669706556 convergence 0.2588762159431427

t, fx, fy = fourbar.solve(ang,l,kl,c,dir,gnd,opt.cs,vis=True)
print(opt.error(t, fx, fy))

fxi = np.interp(opt.td,t,fx)
fyi = np.interp(opt.td,t,fy)

plt.figure()
plt.plot(t,fx,'r')
plt.plot(t,fy,'g')
# plt.plot(opt.td,fxi,'r')
# plt.plot(opt.td,fyi,'g')
plt.plot(opt.td,opt.fxd,'--r')
plt.plot(opt.td,opt.fyd,'--g')
plt.show()
