from transformers import AutoModelForCausalLM

from .models import ARCHITECTURE_ALIAS


def get_deep_attr(obj, attr_path):
    for attr in attr_path.split("."):
        if "[*]" in attr:
            # Handle list-like attributes
            obj = getattr(obj, attr[:-3])[0]
        else:
            obj = getattr(obj, attr)
    return obj


def add_alias_to_attrs(model):
    if type(model).__name__ in ARCHITECTURE_ALIAS:
        alias_dict = ARCHITECTURE_ALIAS[type(model).__name__]

        for alias_path, target_path in alias_dict:
            alias_parts = alias_path.split(".")
            target_parts = target_path.split(".")

            assert len(alias_parts) == len(target_parts), (
                "Alias and target paths must have the same number of parts"
            )

            # Determine where to set the property
            if len(alias_parts) == 1:
                # Top level: set on type(model)
                cls = type(model)
                alias_name = alias_parts[0]
                alias_target = target_parts[0]
                owner_obj = model
            else:
                # Deeper level: walk through model to reach correct sub-object
                parent_path = ".".join(alias_parts[:-1])
                owner_obj = get_deep_attr(model, parent_path)
                cls = type(owner_obj)
                target_parent_path = ".".join(target_parts[:-1])
                assert get_deep_attr(model, target_parent_path) is owner_obj, (
                    "Target path must match owner object"
                )
                alias_name = alias_parts[-1]
                alias_target = target_parts[-1]

            # Check if already defined
            if hasattr(cls, alias_name):
                # print(f"Alias {alias_name} already exists in {cls.__name__}, skipping.")
                continue

            # Define property
            setattr(
                cls,
                alias_name,
                property(lambda self, target=alias_target: get_deep_attr(self, target)),
            )

    return model


class AutoModelForCausalLMWithAliases(AutoModelForCausalLM):
    @staticmethod
    def from_pretrained(model_name_or_path: str, **kwargs) -> type:
        """
        Returns the appropriate universal causal language model class based on the model name or path.

        Args:
            model_name_or_path (str): The name or path of the model.

        Returns:
            class: The appropriate universal causal language model class.
        """
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model = add_alias_to_attrs(model)

        return add_alias_to_attrs(model)
