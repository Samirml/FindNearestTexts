import torch
from transformers import AutoTokenizer
tokenizer_name="bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)

@torch.inference_mode()
def extract_keywords(text: str, device: torch.device, model: torch.nn.Module) -> list:
    """
     Extracts keywords from the given text using a trained NER model.

     Args:
         text (str): Input text to extract keywords from.
         device (torch.device): Device to perform inference on (CPU/GPU).
         model (torch.nn.Module): Trained model for extracting keywords.

     Returns:
         list: List of extracted keywords from the text.
     """
    input_ids = tokenizer([text], return_tensors='pt')['input_ids']
    logits = model(input_ids.to(device)).cpu()
    labels = logits.argmax(-1).squeeze(0)

    entities = []
    entity = []
    for i in range(1, len(labels) - 1):
        if labels[i] == 0:
            if len(entity) != 0:
                entities.append(entity)
            entity = []
        elif labels[i] % 2 == 1:
            if len(entity) != 0:
                entities.append(entity)
            entity = [input_ids[0, i].item()]
        else:
            if i > 0 and labels[i - 1] not in (labels[i], labels[i] - 1):
                if len(entity) != 0:
                    entities.append(entity)
                entity = []
            entity.append(input_ids[0, i].item())
    return tokenizer.batch_decode(entities)

