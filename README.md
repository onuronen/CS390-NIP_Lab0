Report:
The outputs in predictions of custom net is transformed to one-hot encoded arrays for comparing with ytest.
The 2 layer custom net has been implemented with backpropagation, however, the issue is that 
the max index is always 0 in the arrays of predictions. Therefore, only the ytest arrays that 
is of form [1,0,0,.....] matches with the predictions, resulting in 9.8 accuracy. The f-1 score
and backpropagation has been implemented according to slides. I was able to get the accuracy to 98.21
after 40 epochs in tensorflow net. Yet, doubling the epochs is likely to result in 99.
Sigmoid is used for activation in custom net. The skeleton code is slightly changed for tf_net.

