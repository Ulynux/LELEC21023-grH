# Project goals

## Telecommunication
- [x] Change modulation index "h" and proove its optimality 
- [x] Use different values of h (1,2,3,... => int values) and explain why one is better than the others
- [x] Change cutoff frequency
- [x] Prooving that CFO is a bottleneck
- [x] CFO iterative
- [x] Prooving that STO is a bottleneck (simulation)
- [x] STO using Savgol and higher derivatives
- [ ] Prooving that STO is a bottleneck (real measurement en optimal value)
- [x] See what can be done with encoding (see LELEC2880)
- [ ] Implement convolutional encoding in C
- [ ] AGC 

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
- [ ] Jumper 1.8 V à demander au tuteur si c'est aussi possible de faire pour le MCU et pas juste pour le AFE (S11)
- [ ] Panneau solaire (S11)
