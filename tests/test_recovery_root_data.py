import torch
from experiments.lib.recovery_root_data import restrict_to_roots


def test_root_filter_keeps_only_original_training_rows_and_labels():
    train = {'example_id': torch.tensor([10, 11, 12, 13]), 'input_ids': torch.tensor([[i, i] for i in range(4)]),
             'attention_mask': torch.ones(4, 2, dtype=torch.long), 'labels': torch.tensor([0, 1, 1, 0])}
    validation = {'example_id': torch.tensor([10]), 'input_ids': torch.tensor([[100, 100]]),
                  'attention_mask': torch.ones(1, 2, dtype=torch.long), 'labels': torch.tensor([1])}
    data = {'train': train, 'selection': validation, 'evaluation': validation}
    metadata = {'counts': {'train': 4, 'selection': 1, 'evaluation': 1}, 'visible_length_mean': {},
                'indices': {'train': [0, 1, 2, 3], 'selection': [4], 'excluded_evaluation_overlap': []}}
    alignment = {'glue_training_example_ids': [10, 13], 'selected_rows': 2, 'external_labels': 'must not be read'}
    data, metadata = restrict_to_roots(data, metadata, alignment)
    assert data['train']['example_id'].tolist() == [10, 13]
    assert data['train']['labels'].tolist() == [0, 0]
    assert data['train']['input_ids'].tolist() == [[0, 0], [3, 3]]
    assert data['selection'] is validation
    assert metadata['counts']['train'] == 2
    assert not set(metadata['indices']['train']) & set(metadata['indices']['selection'])
