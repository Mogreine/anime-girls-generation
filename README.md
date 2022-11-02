# Anime Girls Generation
Learning diffusion models through anime girls generation

## DDPM
First results:

![manually written code](./data/results/ddpm/crop.png)
![manually written code2](./data/results/ddpm/samples.png)

### Notes
* Images are quite ok
* Training was not stable. For some reason crucial to tune batch size properly -- with 128 I got the results, but with bs 64 or 512 loss suddenly went to nans.
* Got some warnings: "Grad strides do not match bucket view strides."
* 