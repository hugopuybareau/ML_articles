# Context

They create a new network architecture based solely on attention mechanisms.

Trained on several GPUs, it got better score on basic traduction tasks than all the other models.

NNs usually has a sequential nature that precludes the parallelization within training examples. This leads to memory constraints most of the time. However, Transformer is made to avoid recurrence and rely on an attention mechanism. This allows more parallelization by completly relying on an Attention mechanism(setting weights on word within a sequence). There is absolutly no use of sequence-aligned RNNS or convolution. 

# Model architecture

Uses the encoder-decoder structure : 
    - encodes the input sequence into a hidden structure.
    - decodes the hidden structure to generate the output sequence
The attention mechanisme helps the decoder focus on different parts of the input during the output generation.

Encoder : 6 identical layers, within which there is 2 sub-layers. First is Self-attention mechanisme, second is fully connected feed-forward network.

Decoder : 6 identical layers. Same as encoder + a third layer with multi-head attention. The queries, keys and values are linearly projected h times. 

The model uses the 'Scaled Dot_product Attention'. Special formula with a softmax...

# Training

Used AdamOptimizer with a changing learning rate. Increase on the warmup training steps and decrease it slowly after. 

# Regularization

Residual dropout on the output of each sublayer. Also on the sums of the embeddings and positional encoding in both the encoder and decoder stacks. 

Label smoothing, hurts perplexity. Improves accuracy and BLEU score.

Used Beam search at a moment but I did not really get in what extent. Understood that beam search is more efficient than greedy search as is compute several probabalities at the same time. 

# To remember
Transformer is the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
