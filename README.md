# mediapipe face detection Cog model

This is an implementation of the [mediapipe's face detection](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

## Basic usage

```bash
    cog predict \
  -i images=@path/to/file \
  -i blur_amount=1.5 \
  -i bias=0 \
  -i output_transparent_image=true
```

or without cloning the git repo:

```bash
cog predict r8.im/chigozienri/mediapipe-face@latest \
  -i images=@path/to/file \
  -i blur_amount=0.0 \
  -i bias=10 \
  -i output_transparent_image=false
```

`images` (required) can be a path to a png/jpg/jpeg, or zip/tar of multiple png/jpg/jpeg
`blur_amount` is any float >= 0, the amount of blur applied to the edges of the mask
`bias` is an int between 0-255, how light the background of the mask should be (if you want to let some of the original background through)
`output_transparent_image` is a bool, outputs an RGBA of the original image with the mask on the alpha channel if true, or a grayscale image of the mask if false
