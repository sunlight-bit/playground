# Neural Network Playground

The Neural Network Playground is an interactive visualization of neural networks, written in TypeScript using d3.js. Please use GitHub issues for questions, requests, and bugs. Your feedback is highly appreciated!

## Modifications from [original repo](https://dcato98.github.io/playground)
- added Ant activation function
## ANT activation function
This repository contains the Ant for the paper [Neuron signal attenuation activation mechanism for deep learning](https://www.cell.com/patterns/fulltext/S2666-3899(24)00289-7).
## Introduction
Signal activation in neurons, whereby intricate biochemical pathways and signaling mechanisms can produce different outputs from subtle changes, constitute complex systems to study. The output state varies based on activity patterns, reflecting plasticity. From neuroplasticity, neuron signal activation resembles a “black-box” core with dynamic input-output behavior, integrating and transforming signals from other neurons with varying transmission patterns. Although the neural mechanisms underlying this dynamic behavior are not fully understood, they reflect a wide range of neuronal properties. We test this hypothesis from a generalized linear system perspective by constructing computational models of neuronal activation, which enhances our understanding of the signal-processing properties of generalized neurons. Our results demonstrate strong performance across various neural network architectures.
<img width="986" height="1020" alt="image" src="https://github.com/user-attachments/assets/385468a5-0443-49d1-aa4b-52060aad130e" />

## Pytorch Implementation
import torch
import torch.nn as nn

class Ant(nn.Module):
    """Signal attenuation function: x * exp(-|x|/t)"""    
    def __init__(self, t=6.0, mode="both"):
        """
        Args:
        - t: float, attenuation rate (default: 6.0)
        - mode: str, optional "both"|"pos"|"neg" (default: "both")
        """
        super().__init__()
        self.t = t
        self.mode = mode
        
    def forward(self, x):
        # Core computation
        y = x * torch.exp(-torch.abs(x) / self.t)
        
        # Apply mode
        if self.mode == "pos":
            y = torch.where(x >= 0, y, torch.zeros_like(y))
        elif self.mode == "neg":
            y = torch.where(x <= 0, y, torch.zeros_like(y))
        
        return y
    
    def extra_repr(self):
        return f"t={self.t}, mode={self.mode}"

## TensorFlow Implementation
import tensorflow as tf
from tensorflow.keras.layers import Layer
class Ant(Layer):
    """Signal attenuation function: x * exp(-|x|/t)"""    
    def __init__(self, t=6.0, mode="both", **kwargs):
        """
        Args:
        - t: float, attenuation rate (default: 6.0)
        - mode: str, optional "both"|"pos"|"neg" (default: "both")
        """
        super(Ant, self).__init__(**kwargs)
        self.t = float(t)
        self.mode = mode        
    def call(self, inputs):
        # Core computation
        y = inputs * tf.exp(-tf.abs(inputs) / self.t)        
        # Apply mode
        if self.mode == "pos":
            y = tf.where(inputs >= 0, y, tf.zeros_like(y))
        elif self.mode == "neg":
            y = tf.where(inputs <= 0, y, tf.zeros_like(y))        
        return y    
    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(Ant, self).get_config()
        config.update({
            "t": self.t,
            "mode": self.mode
        })
        return config    
    def compute_output_shape(self, input_shape):
        """Output shape is same as input shape"""
        return input_shape
        
## Citation：
Jiang W , Yuan H , Liu W .Neuron signal attenuation activation mechanism for deep learning[J].Patterns, 2025, 6(1),101117.DOI:10.1016/j.patter.2024.101117.

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser

For a faster edit-refresh cycle when developing, run `npm run serve-watch`

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`
