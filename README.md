# Pretraining GPT2 model on Basque

Pretraining code for GPT2 model on Basque language. Includes code to train a tokenizer and a model on various Basque datasets and GPT2 models of different sizes.

## Data

There are three choices for pretraining data: Euscrawl, CC100 and mC4. You can add more options.

## Tokenizer

The tokenizer is trained on the pretraining data. The tokenizer is trained using the `tokenizer/train_tokenizer.py` script. The script takes the HuggingFace tokenizer name and dataset names as input. For example, if you want to train the base GPT2 model tokenizer, you can run the following command.

```bash
python3 train_tokenizer.py gpt2 HiTZ/euscrawl
```

The script `tokenizer/train_tokenizer.slurm` trains a tokenizer for each dataset option. You can modify it to train a tokenizer for other models and datasets.
## Model

The model is trained on the pretraining data. The model is trained using the `model/run_clm.py` script. The script takes a config file as input. The config file contains the model name, tokenizer name, dataset names, and other hyperparameters. For example, if you want to train the base GPT2 model on Euscrawl, you can run the following command.

```bash
torchrun --standalone --nproc_per_node=4 run_clm.py ../configs/gpt2-eus-euscrawl.yaml
```

There are some example configurations in `configs` directory and some scripts in `model` directory. For example, the script `gpt2-eus-euscrawl.slurm` trains a base gpt model on Euscrawl. 

## Example

You can find a base tokenizer and model trained with these scripts on https://huggingface.co/HiTZ/gpt2-eus-euscrawl.