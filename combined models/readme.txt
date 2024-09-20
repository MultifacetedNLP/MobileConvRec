Combined Model 1 applies cross-attention to the last hidden states of both models.
Combined Model 2 applies cross-attention in every layer.

Trained Models Link: https://drive.google.com/file/d/1LD05LxiFIEUPd0zuHqvfxo3jI3eKd-L8/view?usp=sharing
Trained Models file has 2 models (ver1 and ver2)
These models can also be trained in previous_interactions_model.ipynb
2 different models are used to match vocabulary and sequence length with the other model
with which cross-attention is used.

Combined Model 1 guide:
This model uses 2 models.
Model1 is the ver1 model in provided trained models.
Model2 is the gpt2_recommender model in baseline.
Generation cannot handle batches for now so you have to do testing on batch_size = 1

Combined Model 2 guide:
This model uses 2 models.
Model1 is the ver1 model in provided trained models.
Model2 is the gpt2_recommender model in baseline.
Generation cannot handle batches for now so you have to do testing on batch_size = 1

Combined Model 2 candidate guide:
This model uses 2 models.
Model1 is the ver2 model in provided trained models.
Model2 is the gpt2_recommender_with_suggestions model in baseline.
Generation cannot handle batches for now so you have to do testing on batch_size = 1

