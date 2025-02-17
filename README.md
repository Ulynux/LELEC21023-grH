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
- [ ] Record dataset -> 2 methods : either with impulse repsonse with a crack sound either with uart just listen to 1h of 1 sound and segment it into 1 s length audio or 2 , 3 ,... and label it (most importantly)
- [ ] Looking for another classifier (either CNN or RF)
- [ ] Parametrize PCA so that we know that it has learned the right number of axis where we project PC
- [ ] Finally parametrize the chosen classifier with Grid search
- [ ] Bench the performances onto a new dataset and look at generalization 

## Signal Processing
- [ ] PCA should be the main tool we use firstly

## MCU
- [x] First optimization of the clock of our MCU (3MHz)
- [ ] Use the low power mode run and the low power mode sleep but need to reduce the sammpling frequency to max 8 kHz
- [ ] Use the AES accelerator
- [ ] Optimize the fast mult function in spectrogram.c
