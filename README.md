# quantum-friendly-transformer

This repo serves as an initial test of quantum-friendly-transformer where we spectral-normalized layers of transformers to understand how the block encoding affects model performance

After setting up the environment with (you may need to ensure your conda installation):

```
bash setup.sh
```

To train an one-layer transformer on the non-tata promoter dataset, use:

```
python -m src.trainer.train_genomic_bench > ./logging/nontata_vanilla.txt
```

To fine-tune the transformer on the same dataset with specific layer spectral-normalized, use:

```
python -m src.trainer.sn_model_genomic_bench > ./logging/nontata_sn.txt
```
