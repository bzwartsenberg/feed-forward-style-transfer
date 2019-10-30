# feed-forward-style-transfer
Feed-forward neural style transfer


This repository implements feedforward neural style transfer as described in Arxiv:1603.08155v1, using  https://github.com/jayanthkoushik/neural-style/blob/master/neural_style/fast_neural_style as a reference.

This implementation uses Keras with a Tensorflow backend


Files: 

 * **custom_layers.py**: Reflection padding, instance normalization, and residual blocks
 * **losses.py**: Implementation of the Gramm and content losses using VGG-19 network pretrained on imagenet
 * **preprocessing.py**: data preprocessing code, image scaling and loading.
 * **styletransfer.py**: StyleTransfer class implementing the network, training, and logging code.
 * **feed_forward_neural_style_evaluate.ipynb**: Example of using a trained network to transform images
