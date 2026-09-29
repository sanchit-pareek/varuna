# Quarantined checkpoint: v1, oil biased

`oil_unet_v1_oilbiased.pt` scores IoU_oil 0.857 on the Zenodo validation tiles
and is still useless on a real scene. On a clean-water control it predicts oil
with a median softmax of 0.953, covering 99.85 percent of the image.

Two causes, both in how the training set and loss were configured:

1. Class weights were `[0.4, 1.0, 2.5]`. Oil was boosted 2.5x while sea was cut
   to 0.4x, a 6.25x bias toward oil. Those weights make sense when oil is rare
   in training. It was not: tiles are selected to contain at least 50 oil
   pixels, so oil was 19.8 percent of training pixels.
2. No plain-sea tiles. Look-alike chips filled the class-0 budget first, so
   `Images/No oil` contributed nothing and the model never saw calm open water.

Kept for comparison against the retrained model. Do not ship it.
