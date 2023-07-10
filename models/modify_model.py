from .lora import modify_with_lora

modifier_dict = {
    "lora": modify_with_lora,
}


def modify_transformer(transformer, config):
   
    if config.model_modifier:
        if config.model_modifier in modifier_dict:
            transformer = modifier_dict[config.model_modifier](transformer, config)
        else:
            raise ValueError(f"Model modifier '{config.model_modifier}' not found.")

    return transformer
