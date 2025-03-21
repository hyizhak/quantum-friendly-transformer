# quantum-friendly-transformer

This repo serves as an initial test of quantum-friendly-transformer where we spectral-normalized layers of transformers to understand how the block encoding affects model performance

After setting up the environment with (you may need to ensure your conda installation):

```
bash setup.sh
```

To train an one-layer transformer on the [non-tata promoter dataset](https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8)(0-1 classification task), use:

```
python -m src.trainer.train_genomic_bench > ./logging/nontata_vanilla.txt
python ./demo/draw_training_process.py ./logging/nontata_vanilla.txt
```

To fine-tune the transformer on the same dataset with specific layer spectral-normalized, use:

```
python -m src.trainer.fine_tune_sn_model_genomic_bench > ./logging/nontata_sn.txt
python ./demo/draw_training_process.py ./logging/nontata_sn.txt
```

To train an one-layer transformer on the [conll 2003 dataset](https://huggingface.co/datasets/eriktks/conll2003)(POS classification task), use:

```
python -m src.trainer.train_conll2003 > ./logging/conll03_vanilla.txt
python ./demo/draw_training_process.py ./logging/conll03_vanilla.txt
```

To fine-tune the transformer on the same dataset with specific layer spectral-normalized, use:

```
python -m src.trainer.fine_tune_sn_model_conll2003 > ./logging/conll03_sn.txt
python ./demo/draw_training_process.py ./logging/conll03_sn.txt
```

Line graphs of results are stored under `./demo `