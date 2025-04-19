import ipdb.stdout
import torch
import re
from torch.utils.data import Dataset
import pandas as pd

def bin_to_logic(bin):
    if bin == 0:
        return "No"
    elif bin == 1:
        return "Yes"
    
class CausalData(Dataset):
    def __init__(self, data_pth):
        self.data = pd.read_csv(data_pth)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]['input']
        premise_match = re.search(r"Premise:\s*(.+?)\s*Hypothesis:", prompt, re.DOTALL)
        hypothesis_match = re.search(r"Hypothesis:\s*(.+)", prompt, re.DOTALL)
        premise = premise_match.group(1).strip()
        hypothesis = hypothesis_match.group(1).strip()
        return {"subject": premise,
                "premise": premise_match.group(1).strip(),
                "prompt": f"Question: {premise} Can we deduct the following: {hypothesis}? Just answer 'Yes' or 'No.' Answer:",
                "expect": bin_to_logic(self.data.iloc[idx]['label'])}
