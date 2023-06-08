# DeepCORE: An interpretable multi-view deep neural network model to detect co-operative regulatory elements
Gene transcription is an essential process involved in all aspects of cellular functions with significant impact on biological traits and diseases. This process is tightly regulated by multiple elements that co-operate to jointly modulate the transcription levels of target genes. To decipher the complicated regulatory network, we present a novel multi-view attention-based deep neural network that models the relationship between genetic, epigenetic, and transcriptional patterns and identifies co-operative regulatory elements (COREs). We applied this new method, named DeepCORE, to predict transcriptomes in 25 different cell lines, which outperformed the state-of-the-art algorithms. Furthermore, DeepCORE translates the attention values embedded in the neural network into interpretable information, including locations of putative regulatory elements and their correlations, which collectively implies COREs. These COREs are significantly enriched with known promoters and enhancers. Novel regulatory elements discovered by DeepCORE showed epigenetic signatures consistent with the status of histone modification marks. 

![DeepCore_framework](https://github.com/liliulab/DeepCORE/assets/49846287/11fae6f3-31de-4d39-8045-1c54b7587e08)


The DNN architecture consists of two separate paths representing the genetic view and the epigenetic view. Each path starts with a CNN layer which is then passed to a ReLU function connected to max pooling. It then uses bi-directional long short-term memory (BiLSTM) networks with attention mechanism to capture the short-range and long-range dependencies. The learnt attention of the two views are concatenated and given to a fully connected network to predict gene transcription levels.

![DeepCORE_Fig2](https://github.com/liliulab/DeepCORE/assets/49846287/2b9ab9eb-60b7-4c11-a291-6ca601d0f48a)

## Installation Requirements
- python 2.7
- tensorflow 1.13.1
- numpy
- scikit-learn
- matplotlib


## MIT License

Copyright (c) 2023 liliulab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
