# QLoRA-Fine-tuning-for-Film-Character-Styled-Responses-from-LLM
Code for fine-tuning Llama2 LLM with custom text dataset to produce film character styled responses

## Overview

This code utilised QLoRA parameter efficient fine-tuning techniques to create a tailored Llama2 LLM capable of returning responses in the style of Gandalf from The Lord of the Rings

## Project Structure 

1. get_gandalf_data.py - webscrapes Gandalf text dialogue data from online resources

2. gandalf_dataset.py - creates query/response dataset from gandalf.csv which was generated from webscraped dialogue data

3. hyper_params.py - defines hyperparameters for training loop
   
4. train_gandalf.py - fine-tunes base Llama2 model with custom gandalf dataset using QLoRA peft techniques

5. evaluate.py - loads fine-tuned Llama2 model and produces Gandalf style response to input text prompt 

## Author 

Louis Chapo-Saunders
