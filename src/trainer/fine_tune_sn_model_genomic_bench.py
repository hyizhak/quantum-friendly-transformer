import torch
import torch.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters
from genomic_benchmarks.data_check import info
from sklearn.metrics import f1_score, accuracy_score
from genomic_benchmarks.data_check.info import labels_in_order

from spectral_norm_transformer.spectral_normalized_transformer_block import SpectrallyNormalizedTransformerForSequenceClassification
from src.util import LetterTokenizer, build_vocab, token_to_idx, coll_factory, manual_seed

# Set the seed
manual_seed(42)

# Load the dataset
train_dset = HumanNontataPromoters('train', version=0)
test_dset = HumanNontataPromoters('test', version=0)
test_labels = labels_in_order(dset_name='human_nontata_promoters')

print(info('human_nontata_promoters', 0))

# Preprocess the dataset
tokenizer = LetterTokenizer()
vocab = build_vocab(train_dset, tokenizer, use_padding=False)

print(f"Vocab size: {len(vocab)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

collate_fn = coll_factory(vocab, tokenizer, device=device)
train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Model
attn_normalized_model = SpectrallyNormalizedTransformerForSequenceClassification(
    d_model=512, nhead=8, d_ff=2048, num_emb=len(vocab), num_classes=2, max_seq_len=256,
    apply_embedding_sn=False,
    apply_attention_sn=True,
    apply_ffn_sn=False
).to(device)

ffn_normalized_model = SpectrallyNormalizedTransformerForSequenceClassification(
    d_model=512, nhead=8, d_ff=2048, num_emb=len(vocab), num_classes=2, max_seq_len=256,
    apply_embedding_sn=False,
    apply_attention_sn=False,
    apply_ffn_sn=True
).to(device)

sn_models = [attn_normalized_model, ffn_normalized_model]

for model in sn_models:

    model_name = "attn_normalized_model" if model == attn_normalized_model else "ffn_normalized_model"

    model.load_state_dict(torch.load("./model/genomic/vanilla_epoch_15.pth"), strict=False)

    print("=" * 80)
    print(model_name)

    # Initial evaluation
    model.eval()
    test_loader = DataLoader(test_dset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    y_preds = []
    with torch.no_grad():
        for x, y in test_loader:
            y_preds.extend(torch.argmax(model(x), dim=1).tolist())

    metrics = {
        'f1': f1_score(test_labels, y_preds),
        'accuracy': accuracy_score(test_labels, y_preds)
        }

    print(f"epoch: 0, metrics: {metrics}")

    # Fine-tuning
    for param in model.transformer.parameters():
        param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    for epoch in tqdm(range(1, 21)):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            y = y.squeeze()
            loss = criterion(y_pred, y.long())
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), f"./model/genomic/{model_name}_epoch_{epoch}.pth")

        # Evaluation
        model.eval()
        test_loader = DataLoader(test_dset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        y_preds = []
        with torch.no_grad():
            for x, y in test_loader:
                y_preds.extend(torch.argmax(model(x), dim=1).tolist())

        metrics = {
            'f1': f1_score(test_labels, y_preds),
            'accuracy': accuracy_score(test_labels, y_preds)
        }

        print(f"epoch: {epoch}, metrics: {metrics}")
