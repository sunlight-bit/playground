# Neural Network Playground

The Neural Network Playground is an interactive visualization of neural networks, written in TypeScript using d3.js. Please use GitHub issues for questions, requests, and bugs. Your feedback is highly appreciated!

## Modifications from [original repo](https://github.com/tensorflow/playground)
- added ANT activation functions
  
Regarding the ANT activation mechanism, please refer to the following references,
[1] Jiang W , Yuan H , Liu W .Neuron signal attenuation activation mechanism for deep learning[J].Patterns, 2025, 6(1).DOI:10.1016/j.patter.2024.101117.

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser

For a faster edit-refresh cycle when developing, run `npm run serve-watch`

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`
