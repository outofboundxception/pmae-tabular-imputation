from types import SimpleNamespace
import numpy as np
import pandas as pd
from pmae_tabular.data.amputation import generate_missingness
from pmae_tabular.fit_pmae import ProportionalMasker
from pmae_tabular.fit_remasker import ReMasker


def test_pmae_and_remasker_fit_small_matrix():
    rng = np.random.default_rng(0)
    complete = rng.uniform(0, 1, size=(64, 8)).astype(np.float32)
    complete[:, 0] = 0.3  # constant feature exercises safe inverse scaling
    incomplete, mask, _ = generate_missingness(complete, "MCAR", 0.5, 0.25, 10)
    incomplete[:12, 0] = np.nan
    mask[:12, 0] = True
    args = SimpleNamespace(batch_size=16, accum_iter=1, min_lr=1e-5,
                           norm_field_loss=False, weight_decay=0.05, lr=None,
                           blr=1e-3, warmup_epochs=1, embed_dim=16, depth=1,
                           decoder_depth=1, num_heads=4, mlp_ratio=4.0,
                           max_epochs=1, mask_ratio=0.5, encode_func="linear", device="cpu")
    for model_class in (ProportionalMasker, ReMasker):
        model = model_class(args)
        imputed = model.fit(pd.DataFrame(incomplete)).numpy()
        assert imputed.shape == complete.shape
        assert np.isfinite(imputed).all()
        # The trainer must preserve entries that were observed in the input.
        assert np.allclose(imputed[~mask], complete[~mask], atol=1e-6)
        assert np.allclose(imputed[:12, 0], 0.3, atol=1e-6)


def test_legacy_root_imports_remain_available():
    from fit_pMAE import ProportionalMasker as LegacyPMAE
    from fit_ReMasker import ReMasker as LegacyReMasker
    from pMAE import MaskedAutoencoder as LegacyPMAEArchitecture
    from model_mae import MaskedAutoencoder as LegacyReMaskerArchitecture
    from evaluate import evaluator

    assert LegacyPMAE is ProportionalMasker
    assert LegacyReMasker is ReMasker
    assert LegacyPMAEArchitecture is not None
    assert LegacyReMaskerArchitecture is not None
    assert callable(evaluator)
