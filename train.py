import tez
import pandas as pd
from sklearn import model_selection 
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
import numpy as np
#Clase en la que se cargan todos los datos 
#Len devuelve la cantidad de productos que se tienen
#Item devuelde los productos 
class MercadoDataset:
    def __init__(self,prod_id,users, prod_unit_price, rating):
        self.prod_id=prod_id
        self.users=users
        self.prod_unit_price=prod_unit_price
        self.rating=rating
    
    def __len__(self):
        return len(self.prod_id)

    def __getitem__(self, item):
        prod_id= self.prod_id[item]
        users= self.users[item]
        prod_unit_price= self.prod_unit_price[item]
        rating= self.rating[item]

        return{
            "id_Prod": torch.tensor(prod_id, dtype=torch.int),
            "users": torch.tensor(users, dtype=torch.int),
            "prod_unit_prices": torch.tensor(prod_unit_price, dtype=torch.float),
            "ratings": torch.tensor(rating, dtype=torch.int),
        }
#RecSysModel modelo
#Red neuronal de 32 neuronas 
#Linear Entrada de 64 neuronas y salida de una neurona 
#Step scheduler after toma por epocas
#Adam capa de optimizacion(asegura que el error llega al minimo en menos timepo)
class RecSysModel(tez.model):
    def __init__(self, num_prod_ids, num_userss):
        self.num_prod_embed = nn.Embedding(num_prod_ids, 32)
        self.user_embed = nn.Embedding(num_userss, 32)
        self.out = nn.Linear(64, 1)
        self.step_scheduler_after="epoch"
    
    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(),lr= 1e-3)
        return opt
    
    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.7)
        return sch

    def Monitor_de_Metricas(self, output, rating):
        output = output.detach().cpu.numpy()
        output = output.detach().cpu.numpy()
        return {'rmse': np.sqrt(metrics.mean_squared_error(rating, output))}


    def forward(self, prod_ids, userss, prod_unit_prices, ratings=None):
        prodid_embeds = self.prodid_embed(prod_ids)
        user_embeds= self.user_embed(userss)
        prod_unit_embeds= self.prod_unit_embed(prod_unit_prices)
        output = torch.cat([prodid_embeds,user_embeds,prod_unit_embeds],dim=1)
        output = self.out(output) 
        loss = nn.MSEKLoss()(output, ratings.view(-1,1))
        cal_metrics =self.Monitor_de_Metricas(output, ratings.view(-1,1))
        return output, loss, cal_metrics

#
def train():
    df = pd.read_csv("../input/products.csv")
    #prod_id	users	prod_name_long	prod_brand	tags	 prod_unit_price 	Rating
    lbl_prod_id = preprocessing.LabelEncoder()
    lbl_user = preprocessing.LabelEncoder()

    df.prod_id = lbl_prod_id.fit_transform(df.prod_id.values)
    df.user = lbl_user.fit_transform(df.user.values)

    df_train, df_valid=model_selection.train_test_split(
        df, test_size=0.1, ramdom_state=42, stratify=df.rating.values
    )
    train_dataset= MercadoDataset(
        prod_id=df_train.prod_id.values,
        users=df_train.users.values,
        prod_unit_price=df_train.prod_unit_price.values,
        rating=df_train.rating.values
    )
    valid_dataset= MercadoDataset(
        prod_id=df_valid.prod_id.values,
        users=df_valid.users.values,
        prod_unit_price=df_valid.prod_unit_price.values,
        rating=df_valid.rating.values
    )
#tamaño de lote train_bs
    model =RecSysModel(num_prod_ids=len(lbl_prod_id.classes_),num_userss=len(lbl_user.classes_))
    model.fit(train_dataset, valid_dataset, train_bs=1024, valid_bs=1024, fp16=True)

if __name__ =="__main__":
    train()