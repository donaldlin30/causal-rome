import torch
import re
from torch.utils.data import Dataset
import pandas as pd

class CausalData(Dataset):
    def __init__(self, data_pth):
        self.data = pd.read_csv(data_pth)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]['input']
        hypothesis_match = re.search(r"Hypothesis:\s*(.*)", prompt, re.IGNORECASE)
        return {"subject": hypothesis_match.group(1).strip(),
                "prompt": prompt,
                "expect": self.data.iloc[idx]['label']}

if __name__ == '__main__':
    data = CausalData('/home/aditya/workspace/causal-rome/dsets/data.csv')
    print(data[2])

    print('asdad')

