import pandas as pd
import transformers as t
import datasets
import torch


def create_pairs_list(csv_file):
    # Load the DataFrame from the CSV file
    df = pd.read_csv(csv_file)

    # Extract the 'Other Character Lines' and 'Gandalf Lines' columns
    others_lines = df['Other Character Lines'].dropna().tolist()
    gandalf_lines = df['Gandalf Lines'].dropna().tolist()

    # Create a list of tuples from the columns
    pairs_list = list(zip(others_lines, gandalf_lines))

    return pairs_list



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        #define tokeniser
        self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        self.ds = create_pairs_list('gandalf.csv')
        

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx): 
        # create prompt
        TEMPLATE = "Below is something that a person has said to you. Write a response to that person.\n\n### Line:\n{line}\n\n### Response:\n"
        pair = self.ds[idx]
        prompt = TEMPLATE.format(line = pair[0])
        prompt = prompt + pair[1]
        
        #tokenise
        res = self.tokenizer(prompt)
        res["input_ids"].append(self.tokenizer.eos_token_id)
        res["attention_mask"].append(1)
        res["labels"] = res["input_ids"].copy()
        return res

    #calculate max sequence length
    def max_sequence_length(self):
        return max(len(pair[0] + pair[1]) for pair in self.ds)


