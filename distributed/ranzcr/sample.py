import os
import pandas as pd


train_path = "ranzcr/data/train.csv"
train_df = pd.read_csv(train_path)

patients = train_df["PatientID"].unique().tolist()
sample_df = train_df[train_df["PatientID"].isin(patients[:50])]

filenames = [row["StudyInstanceUID"] + ".jpg" for index, row in sample_df.iterrows()]
for filename in os.listdir("ranzcr/data/train"):
    if filename not in filenames:
        os.remove("ranzcr/data/train/" + filename)

sample_df.to_csv("sample.csv", index=False)
