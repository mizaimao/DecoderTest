# Decoder Test

I am trying to implement a text generative AI but in a relatively smaller scale. 

Thanks to this [great post](https://jalammar.github.io/illustrated-gpt2/), the implementation was much smoother than I thought!

## Training
First the configuration must be changed in
```
doe.configs.DefaultConfig  # For training a text generation model.
doe.configs.MIDIConfig     # For training a MIDI-based music generation model.
```
This repo mimics the architecture of GPT-2, and numbers of blocks, heads, embedding dimensions are customizable. Be careful about those sizes, though, as a large number can easily exceed VRAM limit of a common GPU.

Here is what ChatGPT told me about how many layers and attention heads used in each variant:

| Variant           | Number of Layers | Number of Attention Heads |
| ----------------- | -------- | ----- |
| Small     |   12   | 12 |
| Medium    |   24   | 16 |
| Large     |  36    | 20 |
| XL        |  48    | 25 |


Other ways to quickly eat VRAM is embedding dimensions and dictionary size.

Some essential directories should also be created prior to the training.

Training can be launched with
```
python doe/train.py
```

## Generating

The generating script can be called using:
```
python doe/infer.py
```
It's using the same configuration file/object as the training step. It will try to load model weights saved from the training as well.


## Demos

Although it was originally for text generation, I tired to generate music with it. Some of the generated MIDI files can be found in the `demos` folder. 

During the generation stage the first token was given empty so sometimes the first notes are a bit disconnected.

## Next Steps

It's still in its very early development and a lot of user-friendly utilities are not implemented.