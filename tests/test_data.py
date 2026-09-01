from src.data.data_ingestion import load_data


def test_data_loading():

    df = load_data()

    assert df is not None
    assert df.shape[0] == 569
    assert df.shape[1] == 31
    assert "target" in df.columns