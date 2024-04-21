from .modelwrapper import AudioModelWrapper


def get_wrapper(model : str):
    if model == "hilcodec":
        from .hilcodec import ModelWrapper
    elif model == "avocodo":
        from .avocodo import ModelWrapper
    else:
        raise NotImplementedError(f"model '{model}' is not implemented")
    return ModelWrapper
