import torch
from torch.nn.functional import softmax

from transformers import BertTokenizer
from transformers import BertForMaskedLM


if __name__ == "__main__":
    """
    https://github.com/ymcui/Chinese-BERT-wwm
    https://huggingface.co/hfl/chinese-bert-wwm-ext
    """
    pretrained = 'hfl/chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = BertForMaskedLM.from_pretrained(pretrained)

    inputtext = "我爱[MASK]天安门。"

    maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

    input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, masked_lm_labels=input_ids)
    loss, prediction_scores = outputs[:2]
    logit_prob = softmax(prediction_scores[0, maskpos]).data.tolist()
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token, logit_prob[predicted_index])
    """
    的 0.21636968851089478
    """
