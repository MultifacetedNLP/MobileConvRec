from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, List
from fuzzywuzzy import fuzz
import numpy as np
import csv
import evaluate



def is_approximate_substring(substring, string, threshold=70):
    for i in range(len(string) - len(substring) + 1):
        window = string[i:i+len(substring)]
        similarity_ratio = fuzz.ratio(substring, window)
        if similarity_ratio >= threshold:
            return True
    return False

def get_first_five_words(sentence):
    words = sentence.split()  # Split the sentence into a list of words
    return " ".join(words[:10])  # Join the first 5 words back into a string


train_raw = pd.read_json("datasets/amazon_electronics/train.jsonl", lines=True)
valid_raw = pd.read_json("datasets/amazon_electronics/val.jsonl", lines=True)



user_id = []
previous_interactions = []
recommended_product_name = []
turns = []
recommend_indexes = []

for index, row in train_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [(get_first_five_words(product['product_name'])) for product in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_product_name.append(get_first_five_words(row['recommended_product']['product_name']))
    dialog_turns = []
    dialog_index = 0
    found_index = False
    for conv in row['turns']:
        if "COMPUTER" in conv:
            turn = 'computer: '+conv['COMPUTER']
            if (get_first_five_words(row['recommended_product']['product_name']) in turn) and not found_index:
                recommend_indexes.append(dialog_index)
                found_index = True
            dialog_turns.append(turn)
            dialog_index+=1
        if "HUMAN" in conv:
            turn = 'human: '+conv['HUMAN']
            dialog_turns.append(turn)
            dialog_index+=1
    if not found_index: # approximately finding the recommender turn
        for i, dialog_turn in enumerate(dialog_turns):
            if is_approximate_substring(get_first_five_words(row['recommended_product']['product_name']), dialog_turn):
                recommend_indexes.append(i)
                found_index = True
                break
                    
    if not found_index:
        recommend_indexes.append(-1)
                        
    turns.append(dialog_turns)
    
print(len(user_id))
print(len(previous_interactions))
print(len(recommended_product_name))
print(len(recommend_indexes))
df_recommender_train = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_product_name":recommended_product_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_train)}")



user_id = []
previous_interactions = []
recommended_product_name = []
turns = []
recommend_indexes = []

for index, row in valid_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [(get_first_five_words(product['product_name'])) for product in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_product_name.append(get_first_five_words(row['recommended_product']['product_name']))
    dialog_turns = []
    dialog_index = 0
    found_index = False
    for conv in row['turns']:
        if "COMPUTER" in conv:
            turn = 'computer: '+conv['COMPUTER']
            if (get_first_five_words(row['recommended_product']['product_name']) in turn) and not found_index:
                recommend_indexes.append(dialog_index)
                found_index = True
            dialog_turns.append(turn)
            dialog_index+=1
        if "HUMAN" in conv:
            turn = 'human: '+conv['HUMAN']
            dialog_turns.append(turn)
            dialog_index+=1
    if not found_index: # approximately finding the recommender turn
        for i, dialog_turn in enumerate(dialog_turns):
            if is_approximate_substring(get_first_five_words(row['recommended_product']['product_name']), dialog_turn):
                recommend_indexes.append(i)
                found_index = True
                break
                    
    if not found_index:
        recommend_indexes.append(-1)
                        
    turns.append(dialog_turns)
    
print(len(user_id))
print(len(previous_interactions))
print(len(recommended_product_name))
print(len(recommend_indexes))
df_recommender_validation = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_product_name":recommended_product_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_validation)}")




df_recommender_train = df_recommender_train[(df_recommender_train["recommend_indexes"] != -1) & (df_recommender_train["turns"].apply(lambda x: len(x) > 0))]
df_recommender_train['user_id'] = df_recommender_train['user_id'].str.lower()
df_recommender_train['previous_interactions'] = df_recommender_train['previous_interactions'].str.lower()
df_recommender_train['recommended_product_name'] = df_recommender_train['recommended_product_name'].str.lower()
df_recommender_train['turns'] = df_recommender_train['turns'].apply(lambda x: [s.lower() for s in x])


df_recommender_validation = df_recommender_validation[(df_recommender_validation["recommend_indexes"] != -1) & (df_recommender_validation["turns"].apply(lambda x: len(x) > 0))]
df_recommender_validation['user_id'] = df_recommender_validation['user_id'].str.lower()
df_recommender_validation['previous_interactions'] = df_recommender_validation['previous_interactions'].str.lower()
df_recommender_validation['recommended_product_name'] = df_recommender_validation['recommended_product_name'].str.lower()
df_recommender_validation['turns'] = df_recommender_validation['turns'].apply(lambda x: [s.lower() for s in x])






test_raw = pd.read_json("datasets/amazon_electronics/test.jsonl", lines=True)
user_id = []
previous_interactions = []
recommended_product_name = []
turns = []
recommend_indexes = []

for index, row in test_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [(get_first_five_words(product['product_name'])) for product in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_product_name.append(get_first_five_words(row['recommended_product']['product_name']))
    dialog_turns = []
    dialog_index = 0
    found_index = False
    for conv in row['turns']:
        if "COMPUTER" in conv:
            turn = 'computer: '+conv['COMPUTER']
            if (get_first_five_words(row['recommended_product']['product_name']) in turn) and not found_index:
                recommend_indexes.append(dialog_index)
                found_index = True
            dialog_turns.append(turn)
            dialog_index+=1
        if "HUMAN" in conv:
            turn = 'human: '+conv['HUMAN']
            dialog_turns.append(turn)
            dialog_index+=1
    if not found_index: # approximately finding the recommender turn
        for i, dialog_turn in enumerate(dialog_turns):
            if is_approximate_substring(get_first_five_words(row['recommended_product']['product_name']), dialog_turn):
                recommend_indexes.append(i)
                found_index = True
                break
                    
    if not found_index:
        recommend_indexes.append(-1)
                        
    turns.append(dialog_turns)
    
print(len(user_id))
print(len(previous_interactions))
print(len(recommended_product_name))
print(len(recommend_indexes))
df_recommender_test = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_product_name":recommended_product_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_test)}")


df_recommender_test = df_recommender_test[(df_recommender_test["recommend_indexes"] != -1) & (df_recommender_test["turns"].apply(lambda x: len(x) > 0))]
df_recommender_test['user_id'] = df_recommender_test['user_id'].str.lower()
df_recommender_test['previous_interactions'] = df_recommender_test['previous_interactions'].str.lower()
df_recommender_test['recommended_product_name'] = df_recommender_test['recommended_product_name'].str.lower()
df_recommender_test['turns'] = df_recommender_test['turns'].apply(lambda x: [s.lower() for s in x])






model_checkpoint = "gpt2"
bos = '<|startoftext|>'
eos = '<|endoftext|>'
pad = '<|pad|>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, bos_token=bos, eos_token=eos, pad_token=pad, additional_special_tokens=["<|sep|>","computer:", "human:"])

print(len(tokenizer))



import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x



class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_position_embeddings):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).")

        self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

        self.register_buffer("bias", torch.tril(torch.ones((max_position_embeddings, max_position_embeddings), dtype=torch.bool)).view(1, 1, max_position_embeddings, max_position_embeddings), persistent=False)
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

    def _attn(self, query, key, value, attention_mask_q=None, attention_mask_kv=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) #(batch,n_head,seqlen_q,seqlen_kv)
        attn_weights = attn_weights / (value.size(-1) ** 0.5)

        if attention_mask_kv is not None:
            # Attention mask for key-value pairs  # Shape: (batch_size, 1, 1, seq_length_kv)
            attn_weights = attn_weights + attention_mask_kv #attention mask is (0,-inf)

        

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value) #(batch,n_head,seqlen_q,head_dim)
        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask_q: Optional[torch.FloatTensor] = None,
        attention_mask_kv: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.q_attn(hidden_states) 
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim) #(batch,n_head,sequence_len,head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim) #(batch,n_head,sequence_len,head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim) #(batch,n_head,sequence_len,head_dim)
        #attention masks shape (batch,1,1,sequence_len)
        

        attn_output, attn_weights = self._attn(query, key, value, attention_mask_q, attention_mask_kv)
        

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)#(batch,seqlen,n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights



class Block(nn.Module):
    def __init__(self, config, layer_idx,model1,model2):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = GPT2LMHeadModel.from_pretrained(model1).transformer.h[layer_idx].ln_1
        self.ln_2 = GPT2LMHeadModel.from_pretrained(model2).transformer.h[layer_idx].ln_1
        self.attn1 = GPT2LMHeadModel.from_pretrained(model1).transformer.h[layer_idx].attn
        self.attn2 = GPT2LMHeadModel.from_pretrained(model2).transformer.h[layer_idx].attn
        self.ln_3 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_4 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_5 = GPT2LMHeadModel.from_pretrained(model1).transformer.h[layer_idx].ln_2
        self.ln_6 = GPT2LMHeadModel.from_pretrained(model2).transformer.h[layer_idx].ln_2
        self.cross_attn = CrossAttention(hidden_size,config.num_attention_heads,config.max_position_embeddings)
        self.mlp1 = GPT2LMHeadModel.from_pretrained(model1).transformer.h[layer_idx].mlp
        self.mlp2 = GPT2LMHeadModel.from_pretrained(model2).transformer.h[layer_idx].mlp

    def forward(
        self,
        hidden_states1: Optional[Tuple[torch.FloatTensor]],
        attention_mask1: Optional[torch.FloatTensor],
        hidden_states2: Optional[Tuple[torch.FloatTensor]],
        attention_mask2: Optional[torch.FloatTensor],
    ):
        
        residual1 = hidden_states1
        residual2 = hidden_states2
        
        hidden_states1 = self.ln_1(hidden_states1)
        hidden_states2 = self.ln_2(hidden_states2)
        
        attn_outputs1 = self.attn1(
            hidden_states1,
            attention_mask=attention_mask1,
        )
        attn_output1 = attn_outputs1[0]  # output_attn: a, present, (attentions)
        attn_outputs2 = self.attn2(
            hidden_states2,
            attention_mask=attention_mask2,
        )
        attn_output2 = attn_outputs2[0]
        # residual connection
        hidden_states1 = attn_output1 + residual1
        hidden_states2 = attn_output2 + residual2
        


        residual1 = hidden_states1
        residual2 = hidden_states2
        hidden_states1 = self.ln_3(hidden_states1)
        hidden_states2 = self.ln_4(hidden_states2)
        
        cross_attn_output,z = self.cross_attn(hidden_states2,hidden_states1,attention_mask2,attention_mask1)
        hidden_states1 = residual1
        hidden_states2 = cross_attn_output + residual2
        
        
        
        
        
        residual1 = hidden_states1
        residual2 = hidden_states2
        hidden_states1 = self.ln_5(hidden_states1)
        hidden_states2 = self.ln_6(hidden_states2)
        feed_forward_hidden_states1 = self.mlp1(hidden_states1)
        feed_forward_hidden_states2 = self.mlp2(hidden_states2)
        # residual connection
        hidden_states1 = residual1 + feed_forward_hidden_states1
        hidden_states2 = residual2 + feed_forward_hidden_states2

        return hidden_states1,hidden_states2  # hidden_states, present, (attentions, cross_attentions)
    


import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Config
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions




class CrossAttentionGPT2Model(nn.Module):
    def __init__(self, config1,config2,model1,model2):
        super(CrossAttentionGPT2Model, self).__init__()
        self.config1 = config1
        self.config2 = config2
        self.wte1 = GPT2LMHeadModel.from_pretrained(model1).transformer.wte
        self.wpe1 = GPT2LMHeadModel.from_pretrained(model1).transformer.wpe
        self.wte2 = GPT2LMHeadModel.from_pretrained(model2).transformer.wte
        self.wpe2 = GPT2LMHeadModel.from_pretrained(model2).transformer.wpe
        self.drop1 = GPT2LMHeadModel.from_pretrained(model1).transformer.drop
        self.drop2 = GPT2LMHeadModel.from_pretrained(model2).transformer.drop
        self.h = nn.ModuleList([Block(config2,n,model1,model2) for n in range(config1.n_layer)])
        self.ln_f = GPT2LMHeadModel.from_pretrained(model2).transformer.ln_f
        self.lm_head = GPT2LMHeadModel.from_pretrained(model2).lm_head
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        self.dtype = torch.float32

    def forward(self, input_ids1, input_ids2, attention_mask1=None,attention_mask2=None,labels=None):
        # Embed the input tokens using the pre-trained token embeddings
        input_shape1 = input_ids1.size()
        input_ids1 = input_ids1.view(-1, input_shape1[-1])
        batch_size1 = input_ids1.shape[0]
        input_shape2 = input_ids2.size()
        input_ids2 = input_ids2.view(-1, input_shape2[-1])
        batch_size2 = input_ids2.shape[0]
        device = input_ids2.device

        # Get the positional encodings and add to hidden states
        position_ids1 = torch.arange(0, input_shape1[-1], dtype=torch.long, device=device)
        position_ids2 = torch.arange(0, input_shape2[-1], dtype=torch.long, device=device)
        position_ids1 = position_ids1.unsqueeze(0)
        position_ids2 = position_ids2.unsqueeze(0)
        
        input_embeds1 = self.wte1(input_ids1)
        input_embeds2 = self.wte2(input_ids2)
        position_embeds1 = self.wpe1(position_ids1)
        position_embeds2 = self.wpe2(position_ids2)
        hidden_states1 = input_embeds1 + position_embeds1
        hidden_states2 = input_embeds2 + position_embeds2
        
        
        
        attention_mask1 = attention_mask1[:, None, None, :]
        attention_mask2 = attention_mask2[:, None, None, :]
        attention_mask1 = attention_mask1.to(dtype=self.dtype) 
        attention_mask2 = attention_mask2.to(dtype=self.dtype) 
        attention_mask1 = (1.0 - attention_mask1) * torch.finfo(self.dtype).min
        attention_mask2 = (1.0 - attention_mask2) * torch.finfo(self.dtype).min
        
        
        hidden_states1 = self.drop1(hidden_states1)
        hidden_states2 = self.drop2(hidden_states2)
        
        
        output_shape1 = (-1,) + input_shape1[1:] + (hidden_states1.size(-1),)
        output_shape2 = (-1,) + input_shape2[1:] + (hidden_states2.size(-1),)
        

        # Iterate through each custom transformer layer
    
        for i, layer in enumerate(self.h):
            attention_mask1 = attention_mask1.to(hidden_states1.device)
            attention_mask2 = attention_mask2.to(hidden_states2.device)
            out1,out2 = layer(hidden_states1,attention_mask1,hidden_states2,attention_mask2)
            hidden_states1 = out1
            hidden_states2 = out2
            
            
        
        hidden_states2 = self.ln_f(hidden_states2)
        
        hidden_states2 = hidden_states2.view(output_shape2)
        
        hidden_states2 = self.lm_head(hidden_states2)
        
        
        loss = None
        if labels is not None:
            # Shift labels and final_output to the right to align with prediction
            shift_logits = hidden_states2[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        if loss is not None:
            return CausalLMOutputWithCrossAttentions(loss=loss, logits=hidden_states2)
        return CausalLMOutputWithCrossAttentions(logits=hidden_states2)
    
    
    
    
    
    
    
    def get_last_non_padding_token_position(input_ids, pad_token_id):
        # Get the positions of the last non-padding tokens
        non_pad_positions = (input_ids != pad_token_id).nonzero(as_tuple=True)[1]
        last_non_pad_position = non_pad_positions[-1]
        return last_non_pad_position
    
    def generate(self, input_ids1, input_ids2, attention_mask1=None, attention_mask2=None, max_length=20, temperature=1.0, tokenizer=None):
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        # Ensure the model is in evaluation mode
        self.eval()
        if input_ids1.dim() == 1:
            input_ids1 = input_ids1.unsqueeze(0)
        if input_ids2.dim() == 1:
            input_ids2 = input_ids2.unsqueeze(0)
        if attention_mask1 is not None and attention_mask1.dim() == 1:
            attention_mask1 = attention_mask1.unsqueeze(0)
        if attention_mask2 is not None and attention_mask2.dim() == 1:
            attention_mask2 = attention_mask2.unsqueeze(0)

        generated_sequence = []

        with torch.no_grad():
            for _ in range(max_length):
                pos = (input_ids2 != pad_token_id).nonzero(as_tuple=True)[1][-1].item()
                output = self.forward(input_ids1, input_ids2, attention_mask1, attention_mask2)
                logits = output.logits[:, pos, :] / temperature
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                #next_token = torch.multinomial(probabilities, num_samples=1)
                next_token = torch.argmax(probabilities, dim=-1, keepdim=True)

                # Append the generated tokens to the respective sequences
                generated_sequence.append(next_token.item())

                # Break the loop if the EOS token is generated for all sequences
                if (next_token.item() == eos_token_id):
                    break

                # Update input_ids2 and attention_mask2 by appending the new token
                
                if (pos+1==input_ids2.size(1)):
                    input_ids2 = torch.cat([input_ids2[:, 1:], next_token], dim=1)
                    attention_mask2 = torch.cat([attention_mask2[:, 1:], torch.ones((1, 1), device=input_ids2.device)], dim=1)
                else:
                    input_ids2 = torch.cat([input_ids2[:, :pos + 1], next_token,input_ids2[:, pos + 2:]], dim=1)
                    attention_mask2 = torch.cat([attention_mask2[:, :pos + 1], torch.ones((1, 1), device=input_ids2.device),attention_mask2[:, pos + 2:]], dim=1)

        return generated_sequence
    


model1 = 'models/inter_gpt2'
model2 = 'models/conv_gpt2_responder'
config1 = GPT2Config.from_pretrained('models/inter_gpt2')
config2 = GPT2Config.from_pretrained('models/conv_gpt2_responder')
model = CrossAttentionGPT2Model(config1,config2,model1,model2)

model_max_length=512
print(model.dtype)
print(model)



@dataclass
class RecommenderItem:
    prompt: str
    generation: Optional[str] = None
    interaction: Optional[str] = None
    
class recommenderDataset(Dataset):
    def __init__(self, data: List[RecommenderItem]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> RecommenderItem:
        return self.data[idx]
    


items_train = []
for _, row in df_recommender_train.iterrows():
    interactions = bos
    prompt = bos
    if row["previous_interactions"] is not None:
        interactions = interactions + row["previous_interactions"]
    else:
        interactions = interactions + "None"
    for index, turn in enumerate(row["turns"]):
        prompt += turn + "\n"
        if (index + 1) < len(row["turns"]) and index % 2 == 1: # only add the training sample where the computer is speaking
            items_train.append(RecommenderItem(prompt, row["turns"][index + 1] + eos,interactions))


items_validation = []
for _, row in df_recommender_validation.iterrows():
    interactions = bos
    prompt = bos
    if row["previous_interactions"] is not None:
        interactions = interactions + row["previous_interactions"]
    else:
        interactions = interactions + "None"
    for index, turn in enumerate(row["turns"]):
        prompt += turn + "\n"
        if (index + 1) < len(row["turns"]) and index % 2 == 1: # only add the training sample where the computer is speaking
            items_validation.append(RecommenderItem(prompt, row["turns"][index + 1] + eos,interactions))



items_test = []
for _, row in df_recommender_test.iterrows():
    interactions = bos
    prompt = bos
    if row["previous_interactions"] is not None:
        interactions = interactions + row["previous_interactions"]
    else:
        interactions = interactions + "None"
    for index, turn in enumerate(row["turns"]):
        prompt += turn + "\n"
        if (index + 1) < len(row["turns"]) and index % 2 == 1: # only add the training sample where the computer is speaking
            items_test.append(RecommenderItem(prompt, row["turns"][index + 1] + eos,interactions))



def training_collator(batch: list[recommenderDataset]): # for training a language model
    input_ids1 = []
    attention_masks1 = []
    input_ids2 = []
    attention_masks2 = []
    labels = []
    for item in batch:
        interaction_tokens = tokenizer.encode(item.interaction, return_tensors="pt")[0]
        prompt_tokens = tokenizer.encode(item.prompt, return_tensors="pt")[0] 
        generation_tokens = tokenizer.encode(item.generation, return_tensors="pt")[0]
        prompt_len = len(prompt_tokens)
        generation_len = len(generation_tokens)
        interaction_len = len(interaction_tokens)
        unused_len1 = model_max_length - prompt_len - generation_len
        unused_len2 = model_max_length - interaction_len
        # handling case when input is greater than tokenizer length.
        
        if unused_len2 < 0:
            interaction_start_tokens = interaction_tokens[:1]
            trimmed_interaction = interaction_tokens[unused_len2 * -1 + 1 :] # TODO: you could delete the prompt to reach the first |beginuser| token
            interaction_tokens = torch.cat(
                [interaction_start_tokens, trimmed_interaction], axis=0
            )
            prompt_len = len(prompt_tokens)
            unused_len1 = 0
        
        if unused_len1 < 0:
            prompt_start_tokens = prompt_tokens[:1]
            trimmed_prompt = prompt_tokens[unused_len1 * -1 + 1 :] # TODO: you could delete the prompt to reach the first |beginuser| token
            prompt_tokens = torch.cat(
                [prompt_start_tokens, trimmed_prompt], axis=0
            )
            prompt_len = len(prompt_tokens)
            unused_len1 = 0
        pad1 = torch.full([unused_len1], tokenizer.pad_token_id)
        pad2 = torch.full([unused_len2], tokenizer.pad_token_id)
        input_tokens2 = torch.cat(
            [interaction_tokens, pad2]
        )
        input_tokens1 = torch.cat(
            [prompt_tokens, generation_tokens, pad1]
        )
        label = torch.cat(
            [
                torch.full(
                    [prompt_len],
                    -100,
                ),
                generation_tokens,
                torch.full([unused_len1], -100),
            ]
        )
        attention_mask1 = torch.cat(
            [
                torch.full([prompt_len + generation_len], 1),
                torch.full([unused_len1], 0),
            ]
        )
        attention_mask2 = torch.cat(
            [
                torch.full([interaction_len], 1),
                torch.full([unused_len2], 0),
            ]
        )
        input_ids1.append(input_tokens1)
        attention_masks1.append(attention_mask1)
        input_ids2.append(input_tokens2)
        attention_masks2.append(attention_mask2)
        labels.append(label)

    out = {
        "input_ids2": torch.stack(input_ids1),
        "attention_mask2": torch.stack(attention_masks1),
        "input_ids1": torch.stack(input_ids2),
        "attention_mask1": torch.stack(attention_masks2),
        "labels": torch.stack(labels),
    }

    return out



from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="/tmp/models/combined",
    num_train_epochs=10,
    logging_steps=100,
    # logging_dir=self.cfg.logging_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=100,#self.cfg.save_steps,
    eval_steps=100, #self.cfg.eval_steps,
    save_total_limit=4,
    gradient_accumulation_steps=4, #gradient_accumulation_steps,
    per_device_train_batch_size=3, #train_batch_size,
    per_device_eval_batch_size=3, #self.cfg.eval_batch_size,
    warmup_steps=100,
    weight_decay=0.01,
    # dataloader_drop_last=True,
    disable_tqdm=False,
    report_to='none',
    push_to_hub=False
)




trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=recommenderDataset(items_train),
        eval_dataset=recommenderDataset(items_validation), #dm.datasets[DataNames.dev_language_model.value],
        data_collator=training_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )



trainer.train()

torch.save(model.state_dict(), "models/combined/model.pth")

model = model.to(device)

model_checkpoint = "gpt2"
bos = '<|startoftext|>'
eos = '<|endoftext|>'
pad = '<|pad|>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, bos_token=bos, eos_token=eos, pad_token=pad, additional_special_tokens=["<|sep|>","computer:", "human:"],padding_side='right')

print(len(tokenizer))


bleu = evaluate.load("bleu")

def chunk(list_of_elements, batch_size): # using this chunk function, we can split our data to multiple batches
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i:i+batch_size]

def evaluate_responder(dataset, model, tokenizer, batch_size=8, device=device, bleu=bleu):
    prompt_arr = [data.prompt for data in dataset]
    generation_arr = [data.generation for data in dataset]
    interaction_arr = [data.interaction for data in dataset]
    max_length=384
    generation_length=128
    print(len(dataset))
    predictions = []
    references = []
    
    for prompt, generation,interaction in tqdm(zip(prompt_arr, generation_arr,interaction_arr), total = len(generation_arr)):
        
        inputs1 = tokenizer(interaction,max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        inputs2 = tokenizer(prompt,max_length=max_length, truncation=True, padding="max_length", return_tensors="pt") 
        

        generations_predicted = model.generate(input_ids1=inputs1["input_ids"].to(device),input_ids2=inputs2["input_ids"].to(device), attention_mask1=inputs1["attention_mask"].to(device),attention_mask2=inputs2["attention_mask"].to(device),
                            max_length=generation_length,
                            tokenizer=tokenizer) # length_penalty=0.8, Set length_penalty to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.

        generations_predicted = generations_predicted # we only need the generation part, not the prompt part.
        decoded_generation = [tokenizer.decode(generation, clean_up_tokenization_spaces=True).replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")]
        generation = [generation.replace(tokenizer.eos_token, "")]
    
        predictions.extend(decoded_generation)
        references.extend([[gen] for gen in generation])

    results = bleu.compute(predictions=predictions, references=references)
    return results


results = evaluate_responder(recommenderDataset(items_test), model, tokenizer, batch_size=4, device=device, bleu=bleu)
print("results: ", results)



