import torch
from experiments.runners import recovery_matched as matched


def test_matched_policy_uses_all_training_but_no_validation_rows(monkeypatch):
    def data(ids, token_offset=0):
        return {'example_id': torch.tensor(ids), 'input_ids': torch.tensor([[i + token_offset, 9] for i in ids]),
                'attention_mask': torch.ones(len(ids), 2, dtype=torch.long), 'labels': torch.tensor([i % 2 for i in ids])}
    train, selection, evaluation = data([0, 2]), data([1, 3]), data([0, 1], 100)
    monkeypatch.setattr(matched, 'original_prepare_data', lambda *args: (
        None, {'train': train, 'selection': selection, 'evaluation': evaluation},
        {'indices': {'excluded_evaluation_overlap': []}}))
    _, datasets, metadata = matched.prepare_data(None, None, None)
    assert datasets['train']['input_ids'][:, 0].tolist() == [0, 1, 2, 3]
    assert datasets['selection'] is datasets['evaluation']
    assert datasets['selection']['input_ids'][:, 0].tolist() == [100, 101]
    assert metadata['counts'] == {'train': 4, 'selection': 2, 'evaluation': 2}
    assert metadata['selection_equals_evaluation'] is True
    assert not set(metadata['indices']['train']) & set(metadata['indices']['selection'])


def test_length_weighting_changes_only_training_sampling(monkeypatch):
    from experiments.runners import recovery_length_weighted as weighted
    mask = torch.zeros(2, 32, dtype=torch.long)
    mask[0, :2] = 1
    mask[1, :25] = 1
    train = {'example_id': torch.tensor([10, 20]), 'input_ids': torch.tensor([[1] * 32, [2] * 32]),
             'attention_mask': mask, 'labels': torch.tensor([0, 1])}
    evaluation = {'example_id': torch.tensor([0]), 'input_ids': torch.tensor([[100] * 32]),
                  'attention_mask': torch.ones(1, 32, dtype=torch.long), 'labels': torch.tensor([0])}
    metadata = {'counts': {'train': 2, 'selection': 1, 'evaluation': 1},
                'visible_length_mean': {}, 'indices': {'train': [0, 1], 'selection': [2], 'excluded_evaluation_overlap': []}}
    monkeypatch.setattr(weighted.matched, 'prepare_data', lambda *args: (
        None, {'train': train, 'selection': evaluation, 'evaluation': evaluation}, metadata))
    _, datasets, record = weighted.prepare_data(None, None, None)
    assert record['repetitions_per_training_example'] == [1, 3]
    assert datasets['train']['example_id'].tolist() == [10, 20, 20, 20]
    assert datasets['selection'] is evaluation
    assert not (datasets['train']['input_ids'] == 100).any()
    assert record['visible_length_mean']['train'] == 19.25
    assert not set(record['indices']['train']) & set(record['indices']['selection'])
