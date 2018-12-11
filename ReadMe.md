## Future Object Segmentation for Complex Correlated Motions

### Abstract
Predicting the future is an important component of true artificial intelligence, allowing systems to antici-pate events and act accordingly. In computer vision the task is generally formulated as inferring a futurerepresentation from past images or frames.  Practical applications also span a wide range, with modelscapable of predicting the future being a natural fit in contexts such as robotics or autonomous driving.In its richest form, the task consists of rendering full RGB images from past frames.  This has proven asubstantial challenge however, and many alternative formulations attempt to make predictions in a sim-pler and more constrained space.  A recent thread that has gained considerable momentum consists of predicting future semantic masks. We tackle this problem under the light of object segmentation.

We develop models and create datasets that are fit to the challenge, and investigate their behaviour to gaindeeper insights. We first introduce a new data bank, CorrelatedMotions. We argue it is more suited to thetask of future image segmentation than Cityscapes, which is currently often used for benchmarking. Wealso develop a model for inference, which includes a bespoke architecture that we name SpatialNet. Weevaluate this model on CorrelatedMotions, and show it overall outperforms simpler baselines.  Finally,we delve deeper into our model through a series of experiments. We extend the scope of our task to futurepredictions in variable times, and work towards a deeper understanding on the mechanics responsible for the behaviours we observe.

### Citation
For using this work, please provide the following citation:

@misc{FutureObjectValassakis2018, \
author={Valassakis, Pierre Eugene and Brostow, Gabriel},\
title = {Future Object Segmentation for Complex Correlated Motions},\
howpublished={https://github.com/eugval/Motion_Prediction}
}
