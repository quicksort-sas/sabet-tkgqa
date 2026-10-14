from tkg_qa_models.models.lm_baselines import LMBaseline, EmbedKGQA, CronKGQA
from tkg_qa_models.models.tempoqr import TempoQR
from tkg_qa_models.models.sabet import Sabet
from tkg_qa_models.models.sabet1 import Sabet1

MODELS = {
    "bert": LMBaseline,
    "roberta": LMBaseline,
    "embedkgqa": EmbedKGQA,
    "cronkgqa": CronKGQA,
    "tempoqr": TempoQR,
    "subgtr": lambda tkbc_model, args: TempoQR(tkbc_model, args, time_sensitivity=True),
    "sabet": Sabet,
    "sabet1": Sabet1,
}
MODEL_ALIASES = {"tempo_qr": "tempoqr", "imsabt": "sabet"}


def build_model(name, tkbc_model, args):
    name = MODEL_ALIASES.get(name, name)
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](tkbc_model, args)
