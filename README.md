# quantum-friendly-transformer

This repo serves as an initial test of quantum-friendly-transformer where we spectral-normalize / frobenius-normalize layers of transformers to understand how the block encoding affects model performance

After setting up the environment with (you may need to ensure your conda installation):

```
bash setup.sh
```

To train an single-layer transformer on the [Genomic Benchmarks non-tata promoter dataset](https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8)(0-1 classification task), use:

```
python src/quantum_friendly_transformer/trainer/train_genomic_bench.py > ./logging/nontata_vanilla.txt
python demo/draw_training_process.py ./logging/nontata_vanilla.txt
```

To fine-tune the transformer on the same dataset with specific layer normalized, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_sn_model_genomic_bench.py > ./logging/nontata_sn.txt
python demo/draw_training_process.py ./logging/nontata_sn.txt
```

To fine-tune the frobenius normalized DNABert on the same dataset, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_multilayer_genomic_bench.py > ./logging/nontata_multi_layer.txt
```

To train an one-layer transformer on the [Genome Understanding Evaluation (GUE) notata promoter dataset](https://huggingface.co/datasets/leannmlindsey/GUE)(0-1 classification task), use:

```
python src/quantum_friendly_transformer/trainer/train_gue.py > ./logging/gue_vanilla.txt
python demo/draw_training_process.py ./logging/gue_vanilla.txt
```

To fine-tune the transformer on the same dataset with specific layer normalized, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_sn_model_gue.py > ./logging/gue_sn.txt
python demo/draw_training_process.py ./logging/gue_sn.txt
```

To train an one-layer transformer on the [conll 2003 dataset](https://huggingface.co/datasets/eriktks/conll2003)(POS classification task), use:

```
python src/quantum_friendly_transformer/trainer/train_conll2003.py > ./logging/conll03_vanilla.txt
python demo/draw_training_process.py ./logging/conll03_vanilla.txt
```

To fine-tune the transformer on the same dataset with specific layer normalized, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_sn_model_conll2003.py > ./logging/conll03_sn.txt
python demo/draw_training_process.py ./logging/conll03_sn.txt
```

Line graphs of results are stored under `./demo `