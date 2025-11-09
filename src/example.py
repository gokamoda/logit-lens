import torch
from transformers import AutoTokenizer

from logit_lens import (
    AutoModelForCausalLMWithAliases,
    LogitLens,
    LogitLensResult,
)
from logit_lens.visualizers import visualize_layer_results_core


def visualize_layer_results(
    result: LogitLensResult,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
):
    probs = []
    if result.post_emb_probs is not None:
        probs.append(result.post_emb_probs.unsqueeze(0))
    if result.post_layer_probs is not None:
        probs.append(result.post_layer_probs)

    decoded = []
    if result.post_emb_token_ids is not None:
        decoded.append(
            [
                tokenizer.convert_ids_to_tokens(position)
                for position in result.post_emb_token_ids.cpu()
            ]
        )
    if result.post_layer_token_ids is not None:
        decoded.extend(
            [
                [tokenizer.convert_ids_to_tokens(position) for position in layer]
                for layer in result.post_layer_token_ids.cpu()
            ]
        )

    y_labels = []
    if result.post_emb_token_ids is not None:
        y_labels.append("emb.")
    if result.post_layer_token_ids is not None:
        y_labels.extend([f"layer {i}" for i in range(len(result.post_layer_token_ids))])

    # Get the logits and other relevant information from the resul
    # Call the core visualization function
    fig = visualize_layer_results_core(
        probs=torch.cat(probs, dim=0).cpu(),
        inputs=tokenizer.convert_ids_to_tokens(input_ids.cpu()),
        decoded=decoded,
        y_labels=y_labels,
    )
    return fig


def main():
    # model_name = "meta-llama/Llama-3.2-1B"
    model_name = "openai-community/gpt2"
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLMWithAliases.from_pretrained(model_name)

    # Sample input text
    input_text = "Once upon a time"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text
    with (
        torch.no_grad(),
        LogitLens(model, post_emb=True, post_layer=True, topn=5) as ll,
    ):
        model(**inputs)
        ll_result = ll.get_results()

    fig = visualize_layer_results(
        result=ll_result[0],
        input_ids=inputs["input_ids"][0],
        tokenizer=tokenizer,
    )
    fig.write_html("./logit_lens_visualization.html")


if __name__ == "__main__":
    main()
