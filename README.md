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

## Download code
```python
git clone https://github.com/liliulab/DeepCORE
cd DeepCORE
```

## Usage
### input File
To train the DeepCORE model, users are required to provide a tab sepearated file with the following mandatory columns:
- gene_id: Ensembl gene id 
- chromosome_name: Chromosome location of the gene
- sequence: DNA sequence flanking $\pm$ 5000bps (user desiered length) of the transcription start site (TSS) of the gene
- TPM: Gene expresion measured in terms of TPM (Transcripts Per Million)
- chipseq 1-5: Position-specific normalized read counts for 5 histone modification marks (H3K4me1, H3K4me3, H3K9me3, H3K27me3, and H3K27ac, in that particular order). Each column should contain a comma-seperated string with 10000 values (user defined length and should match the length of the DNA sequence).

Additional information through columns can be provided. The following screenshot shows a sample of input file with the required columns:
![Sample Input File](https://github.com/liliulab/DeepCORE/assets/18314073/a7507c87-d20c-49cc-bc1f-ed0cd5aaf247)


### Training DeepCORE model
DeepCORE model can be trained using the DeepCORE_train.py file. It can be run using the following command:

```
python -u DeepCORE_train.py --input_file='/path_to_tab_seperated_input_file' --add_epigenetic_info='True' ----add_sequence_info='True' --save= '/path_to_save_model' > '/path_to_output.txt'
```

The above command runs with the default settings for training. Additional settings that can be included along with the above code are:
* **Input settings**
    * **--epigenetic_index** = Choose between 1, 2, 3, 4, 5, and all representing chipeseq 1-5 or all markers. Users can also choose multiple indices seperated by comma. (default = 'all')
    * **--normalize** = A boolean setting to indicate if data nomralization is needed. (default = False)
    * **--flanking_region** = The default flanking region is $\pm$ 5000bps around TSS. If the user wants to reduce the flanking region, use this setting. Users can choose between 'upstream** 'downstream** 'both** and 'none'. (default = 'none'. This indicates to use all 10000bps region)
    * **--flanking_width** = Use this parameter only if you choose the --flanking_region to be upsteram, downsteram, or both. This setting will indicate the width of flanking around TSS (default='10000')
    * **--task** = This parameter indicates whether the prediction task is regression or classification. (default = 'regression')
    * **--num_classes** = This parameter indicates the number of output labels for a classification task. (default = 1 which corresponds to regression)
    * **--split_percent** = Users can indicate the percent of the samples to be used for training. (default = 0.8 representing 8-% for training and 20% for validation)
    * **--balanced_train** = Boolean parameter for training set to be balanced before training. (default = False)
    * **--chromosome_name** = This is used to select chromosomes that should be included for analysis, comma delimited. (default = 'All')
    * **--bin_cutoff** = This parameter indicates the probability cutoff to decide the class labels. Must be between 0 and 1.. (default = 0.5)
 
* **CNN parameters**
    * **--cnn_num_layers** = Number of CNN layers. (default = 1)
    * **--cnn_filter_sizes** = CNN window size for each CNN layer, comma seperated. Must be equal to --cnn_num_layers (default = '5')
    * **--cnn_filter_sizes** = CNN window size for each CNN layer, comma seperated. Must be equal to --cnn_num_layers (default = '5')
    * **--cnn_stride_length** = Filter strides, comma seperated. Must be equal to --cnn_num_layers (default = '1')
    * **--cnn_pool_sizes** = Pooling length, comma seperated. Must be equal to --cnn_num_layers (default = '50')

* **Attention layer parameters**
    * **--attn_hid_dims** = Number of hidden dimensions for attention layer, comma delimited. (default = '32')
    * **--attn_wt_randomize** = A boolean flag to initialize the attention weights to random. (default = False)
    * **--add_enc_last_state** = Boolean flag to include last state of encoder to attention computation.  (default = False)
    * **--attn_score_activator** = Choose the attention activation function. Users can choose between relu and tanh.  (default = 'tanh')
    * **--attn_estimator** = Parameter to indicate attention estimator function. Users cam choose between softmax, softmax_temp, gumbel_softmax, and sparsemax. (default = 'softmax')
    * **--attn_temp**  = This parameter is used only if --attn_estimator=softmax_temp. This indicates the sharpening temperature. (default = 1.0)
    * **--attn_regularizer** =  Attention regularization function to be used. Choose between l0, l2, entropy, deviation. (default = 'l2')
    * **--attn_reg_lambda** = L2 regularization for attention. (default = 0.005)
    * **--attn_type** = Type of attention: self vs soft. (default = 'soft')

* **Decoder Unit parameters**
    * **--decoder_multi_view** = Flag to set seperate views for sequence and epigenetics. If False, the two views are concatenated and then given to the decoder. (default = False)
    * **--decoder_model_type** = Choose between None, bilstm, biurnn, bigru, bilstm, bilstmp, and fcn for the decoder model. (default='fcn')
    * **--decoder_hid_dims** Number of hidden dimensions for decoder, comma seperated. default='32')

* **Fully Connected Network layer**
    * **--num_fc_layers** = Number of fully connected layers to be used after the decoder layer. (default=1)
    * **--num_fc_neurons** = Number of hidden units for fully connected layers. comma seperated. Must be equal to --num_dc_layers. (default = '32')

* **Other hyperparameters**
    * **--dropout_rate** = This is used to handle overfitting. (default = 0.5)
    * **--randomize_weights** = If true, the FCN weights are randomly initialized. Else, they are initialized by zeros. (default = False)
    * **--seq_random_weights** = If true, the CNN weights for the sequence view are randomly initialized. Else, they are initialized by zeros. (default = False)
    * **--epi_random_weights** = If true, the CNN weights for the epigeneitc view are randomly initialized. Else, they are initialized by zeros. (default = False)

* **Other settings**
    * **--train_epochs** = Number of training epochs. (default=100)
    * **--learn_rate** = Learning rate for the model. (default = 0.001)
    * **--batch_size** = Batch size for training. (default = 100)
    * **--out_reg_lambda** = L2 regularization parameter. (default = 0.005)
    * **--eval_interval** = Evaluation interval. (default = 1)
    * **--save** = This argument specifies the path to save the model generated by the code.
    * **--out_file** = This argument specifies the path to save the performance scores.


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
