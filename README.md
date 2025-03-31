# Project goals

## Telecommunication
- [ ] Change modulation index "h" and proove its optimality 
- [ ] Use different values of h (1,2,3,... => int values) and explain why one is better than the others 
- [x] Prooving that CFO is a bottleneck
- [ ] Prooving that STO is a bottleneck (simulation)
- [ ] Prooving that STO is a bottleneck (real measurement en optimal value)
- [ ] Testing different demodulations methods (discriminateur using simu)
- [ ] See what can be done with encoding (see LELEC2880)
- [ ] Last use of turbocodes (see LELEC2880)

## Classification
- [x] Record dataset -> 2 methods : either with impulse repsonse with a crack sound either with uart just listen to 1h of 1 sound and segment it into 1 s length audio or 2 , 3 ,... and label it (most importantly)
- [x] Looking for another classifier (either CNN or RF)
- [x] Parametrize PCA so that we know that it has learned the right number of axis where we project PC
- [x] Finally parametrize the chosen classifier with Grid search
- [x] Bench the performances onto a new dataset and look at generalization
- [ ] background class (beware to the SNR)
- [ ] Looking for CNN

## Signal Processing
- [ ] Moving Average
- [ ] H(f) micro

## MCU
- [x] First optimization of the clock of our MCU (3MHz)
- [-] Use the low power mode run and the low power mode sleep but need to reduce the sammpling frequency to max 8 kHz (pour le rapport)
- [x] Use the AES accelerator
- [ ] Optimize the fast mult function in spectrogram.c (Pas obligatoire pour S11)
- [ ] Jumper 1.8 V (S11)
- [ ] Panneau solaire (S11)
