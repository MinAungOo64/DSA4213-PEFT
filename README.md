# DSA4213 Assignment 3

HuffPost News Classification — PEFT Experiments  
This project explores parameter-efficient fine-tuning (PEFT) methods — `LoRA`, `IA³`, and `Prompt Tuning` — using the `HuffPost News Category Dataset`. Two setups are compared:  
1. `Generic DistilBERT` (base model without news fine-tuning)  
2. `News-tuned DistilBERT` (further adapted on the HuffPost dataset).

## Dependencies  

`pip install torch torchvision torchaudio`  
`pip install transformers datasets peft evaluate accelerate scikit-learn`

## File	Description  
Main.ipynb:	Main experiments using the generic DistilBERT model (`LoRA`, `IA³`, `Prompt Tuning`).  
Further Experiment.ipynb:	Follow-up experiments using the news-tuned base model and larger datasets.  

## How to Run
1. Open the notebook folder (the dataset downloads automatically from Hugging Face).
2. Run the main experiments using: `jupyter notebook Main.ipynb`.
   This notebook trains and evaluates `LoRA (r = 2, 8, 16)`, `IA³`, and `Prompt Tuning (10, 50, 100 tokens)`.
3. Run the further experiments using: `jupyter notebook 'Further Experiment.ipynb'`.
   This continues with larger datasets, news-tuned base model comparisons, and performance analysis on accuracy and F1 metrics.

## Dataset
Dataset: https://huggingface.co/datasets/heegyu/news-category-dataset

## References
• Hugging Face Transformers – https://huggingface.co/docs/transformers  
• Hugging Face PEFT – https://github.com/huggingface/peft  
• HuffPost News Category Dataset – https://huggingface.co/datasets/heegyu/news-category-dataset


