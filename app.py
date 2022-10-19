import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from utils.PreProcessData import *

model_path = 'C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/models/'
model_name = 'test_model.pt'

@st.cache
def loadmodel():
    chemberta = AutoModel.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
    return chemberta

def loadtokenizer():
    tokenizer = AutoTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
    return tokenizer

def buildmodel():
    model = RoBERTa_and_DNN_script(
            roberta=chemberta,
            one_hot_enc_len=1,
            n_hidden_layers=2,
            layer_sizes=[350,20],
            dropout=0.2,
            activation='ReLU',
            cls_tok=True,
            add_transformer_layer=False)
    #model = load_ckp(model_name, model)
    model.eval()
    return model

class RoBERTa_and_DNN_script(nn.Module):
    '''
    Class to build Transformer and DNN structure. Supports maximum 3 hidden layers and RoBERTa transformers.
    '''
    def __init__(self, roberta, one_hot_enc_len, n_hidden_layers, layer_sizes, dropout, activation, cls_tok, add_transformer_layer):
        super(RoBERTa_and_DNN_script, self).__init__()
        self.roberta = roberta 
        self.one_hot_enc_len = one_hot_enc_len
        self.n_hidden_layers = n_hidden_layers
        self.layer_sizes = layer_sizes
        self.dropout = nn.Dropout(dropout)
        self.cls_tok = cls_tok
        self.add_layer = add_transformer_layer
        
        self.active =  nn.ReLU()

        self.fc1 = nn.Linear(768 + 1 + one_hot_enc_len,  layer_sizes[0]) # This is where we have to add dimensions (+1) to fascilitate the additional parameters
        self.fc2 = nn.Linear(layer_sizes[0],  layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], 1)
        

    def forward(self, sent_id, mask, exposure_duration, one_hot_encoding):

        roberta_output = self.roberta(sent_id, attention_mask=mask)[0]#.detach()#[:,0,:]#.detach() # all samples in batch : only CLS embedding : entire embedding dimension

        roberta_output = roberta_output[:,0,:]
      
        inputs = torch.cat((roberta_output, torch.t(exposure_duration.unsqueeze(0)), one_hot_encoding), 1)

        x = self.fc1(inputs)
        x = self.active(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        x = x.squeeze()
        return x

def load_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


# APP
app_mode = st.sidebar.selectbox('Select Page',['Home','Documentation'])

if app_mode=='Home': 
    st.title('''fishbAIT''')
    st.markdown('A deep learning software that lets you predict chemical ecotoxicity to fish')
    endpoints = {'EC50': 'EC50', 'EC10': 'EC10'}
    effects = {'MOR': 'MOR', 'DVP': 'DVP', 'GRO': 'GRO','POP': 'POP','MPH':'MPH'}
    model_type = {'EC50': 'EC50_only_model','Chronic': 'EC10_NOEC_only_model','Combined model': 'combo_model'}
    
    endpoint = st.sidebar.radio("Select Endpoint ",tuple(endpoints.keys()))
    effect = st.sidebar.radio("Select Effect ",tuple(effects.keys()))
    modeltype = st.sidebar.radio("Select Model type", tuple(model_type.keys()))

    file_up = st.file_uploader("Upload csv data containing SMILES, Duration, Effect & Endpoint", type="csv")
    
    if file_up:
        df=pd.read_csv(file_up, sep=';') #Read our data dataset
        df = PreProcessData(df).GetOneHotEnc(list_of_endpoints=[endpoint], list_of_effects=['MOR'])
        st.write(df.head())

    if st.button("Predict"):
        with st.spinner(text = 'Inference in Progress...'):
            chemberta = loadmodel()
            tokenizer = loadtokenizer()
            model = buildmodel()
            enc =  tokenizer.batch_encode_plus(df.SMILES.tolist(),
            padding='longest',
            return_tensors='pt')
            with torch.no_grad():
                out = np.round(model(
                        enc['input_ids'],
                        enc['attention_mask'], 
                        torch.from_numpy(np.log10(df.Duration.tolist())).float(), 
                        torch.tensor(df.OneHotEnc_concatenated.tolist())
                        ).numpy(),2)
            st.balloons()
            result = df.copy()
            st.success(f'Predicted effect concentration(s):')
            result['Predictions [Log10(mg/L)]'], result['Predictions [mg/L]'] = out.tolist(), (10**out).tolist()
            st.write(result.head())




