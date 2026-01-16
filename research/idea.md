# Give better performance for training

- Pass L,Edge and gaussian matrix instead of the classics RGB pictures ( no testing)
- Pass a filter with full 255 value pixel as mask to force the model to learn what is a good transistor (null)
- Changing the loss function so that we can add a contraint on the loss and reduce the option (RMSE : X = f(x) idendité ce qui est facile a faire) SSIM + SSB
- Noising the pictures so that the idendity function is no more
