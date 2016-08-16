# NoiseGen - VapouSynth Noise Generator #

*   use uniform/normal distribution to generate random number as noise

## Build ##

*   compiler with c++11 support

## Usage ##

    noisegen.Generate(src, str=1.0, limit=128.0, dyn=True, planes=[0])

*   the default setting for GRAY/YUV source, generate noise pattern on luma only
***

    noisegen.Generate(src, str=1.0, limit=128.0, dyn=True, planes=[0, 1, 2])

*   the default setting for RGB source, generate noise pattern on all planes
***

    noisegen.Generate(src, dyn=False)

*   use static noise pattern for each frame
***

    noisegen.Generate(src, str=1.5, limit=128.0, var=4.0)

*   higher value of strength and var will produce more noticeable noise pattern
***

    noisegen.Generate(src, type=1, mean=0.0, var=1.0)

*   mean and var stands for mean and radius for uniform distribution, generate number within [mean-var, mean+var]
***

    noisegen.Generate(src, type=2, mean=0.0, var=1.0)

*   mean and var stands for mean and variance for normal distribution
***

## Parameter ##

    noisegen.Generate(clip clip[, float str=1.0, float limit=128.0, int type=2, float mean=0.0, float var=1.0, bint dyn=True, bint full=False, int[] planes=[0]])

*   clip: the input clip
    *   all formats support

***
*   str: 'multiplier' for the value of random number
    *   default: 1.0
    *   range: 0.0 ... 128.0

***
*   limit: hard 'limitation' for pixel change considered in 8 bit
    *   default: 128.0
    *   range: 0.0 ... 128.0
    *   note: only works for type=2

***
*   type: noise type
    *   default: 2
    *   1: uniform distribution
    *   2: normal distribution

***
*   mean: control the 'mean' for uniform/normal distribution
    *   default: 0.0
    *   range: -inf ... inf

***
*   var: control the 'radius' for uniform distribution/the 'variance' for normal distribution
    *   default: 1.0
    *   range: 0.001 ... inf

***
*   dyn: use the dynamic noise pattern for each frame if set
    *   default: True

***
*   full: indicate the input is full range (aka pc range) or not, only works for integer type input now.
    *   default: False for vs.GRAY, vs.YUV, vs.YCOCG
                 always True for vs.RGB

***
*   planes: generate noise for planes specified by this param
    *   default: [0] for vs.GRAY, vs.YUV, vs.YCOCG.
                 [0, 1, 2] for vs.RGB.

***

## License ##

    NoiseGen - VapouSynth Noise Generator

    DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
    Version 2, December 2004

    This program is free software. It comes without any warranty, to
    the extent permitted by applicable law. You can redistribute it
    and/or modify it under the terms of the Do What The Fuck You Want
    To Public License, Version 2, as published by Sam Hocevar. See
    http://sam.zoy.org/wtfpl/COPYING for more details.
