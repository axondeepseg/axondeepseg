from unittest.mock import patch

import pytest

from AxonDeepSeg.download_model import (
    SUCCESS_EXIT_CODE,
    get_model_cards,
    main,
    print_available_models,
)


@pytest.mark.unit
def test_model_cards_have_nonempty_unique_training_modalities():
    supported_modalities = {'TEM', 'SEM', 'BF', 'CARS'}
    for name, card in get_model_cards().items():
        modalities = card['modalities']
        assert isinstance(modalities, list), name
        assert modalities, name
        assert all(isinstance(modality, str) for modality in modalities), name
        assert len(modalities) == len(set(modalities)), name
        assert set(modalities) <= supported_modalities, name


@pytest.mark.unit
@pytest.mark.parametrize('model_name', ['multires-TEM', 'unmyelinated-TEM'])
def test_tem_models_do_not_advertise_other_training_modalities(model_name, capsys):
    card = get_model_cards()[model_name]
    assert card['modalities'] == ['TEM']

    print_available_models({model_name: card})

    modality_line = next(line for line in capsys.readouterr().out.splitlines()
                         if line.startswith('Modalities'))
    assert modality_line.split() == ['Modalities', 'TEM']


@pytest.mark.unit
@pytest.mark.parametrize('modalities, expected', [
    (['TEM', 'SEM', 'BF', 'CARS'], 'TEM, SEM, BF, CARS'),
    (['BF'], 'BF'),
    ([], 'N/A'),
    (None, 'N/A'),
])
def test_list_models_formats_training_modalities(modalities, expected, capsys):
    card = {
        'n_classes': 2,
        'model-info': 'Test model',
        'training-data': 'Test data',
    }
    if modalities is not None:
        card['modalities'] = modalities

    print_available_models({'test-model': card})

    modality_line = next(line for line in capsys.readouterr().out.splitlines()
                         if line.startswith('Modalities'))
    assert modality_line.split(maxsplit=1)[1] == expected


@pytest.mark.unit
def test_cli_list_shows_training_modalities_without_downloading(capsys):
    with patch('AxonDeepSeg.download_model.download_model') as download:
        with pytest.raises(SystemExit) as error:
            main(['--list'])

    assert error.value.code == SUCCESS_EXIT_CODE
    assert 'TEM, SEM, BF, CARS' in capsys.readouterr().out
    download.assert_not_called()
