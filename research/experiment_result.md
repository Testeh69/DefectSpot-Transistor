# Experiment

## Loss function

RMSE Loss function ---> idendify to weel the function idendity x = f(x) so we need a contrainct
SSIM Loss function ---> same
SSIM + SSD ----> On test

Test 1:

SSIM + SSD

résultat:
    - Ne reconstruit pas assez bien l'image
    - Baisse trop lente de la loss
conclusion: augmenter alpha ou baisser lambda

## Data

Using normal pictures
Using noisy pictures




Making an AE from scratch detect the dommage on transistors but not on pin, maybe a U-net with skip conection to retain the features is neccesary.



Conclusion:


An U-net is used for segmentation task 