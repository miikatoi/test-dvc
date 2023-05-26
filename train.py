import ray
import time

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
from dvc.repo import Repo
import os
import boto3


class Trainer:

    def get_data(self, path='data.csv'):
        df = pd.read_csv(path)

        df['example'] = df.apply(
            lambda x: InputExample(texts=x['texts'], label=x['label']), axis=1
            )

        return df.example.tolist()


    def train(self, model_path='model', data_path='data.csv'):
        

        #Define the model. Either from scratch of by loading a pre-trained model
        model = SentenceTransformer(model_path)

        #Define your train examples. You need more than just two examples...
        train_examples = self.get_data(path=data_path)

        #Define your train dataset, the dataloader and the train loss
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)

        #Tune the model
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
        model.save("output")
        import shutil
        shutil.make_archive("output", 'zip', "output")
        print(model)

    def save_result(self):
        s3 = boto3.client(
            's3',
            endpoint_url="http://192.168.0.33:9000",
            aws_access_key_id='admin',
            aws_secret_access_key='password',
        )
        s3.upload_file("output.zip", "artifacts", "test/model")



t = Trainer()

@ray.remote
def run_training():
    import os

    # Print current working directory
    print("Current Working Directory:")
    print(os.getcwd())

    # List files and directories
    print("\nFiles and Directories:")
    for item in os.listdir():
        print(item)
    # repo = Repo(".")
    # repo.pull()
    t.train()
    t.save_result()


# t.train()
# t.save_result()
ray.init()
ray.get(run_training.remote())
