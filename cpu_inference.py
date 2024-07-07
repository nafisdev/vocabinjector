import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.bfloat16)
# infer
def predict(news,vocab):
    prompt = f"<human>: Rephrase {news} without changing the structure, but use these vocabulary {vocab}\n<bot>:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    return output_str


gist_url = 'https://gist.githubusercontent.com/nafisdev/381aac7ee98f6e4311ae90798f308956/raw/035681ab33e53cbd09c30c6cb59827d06cc8bd72/vocab.txt'


response = requests.get(gist_url)
gist_content = response.text
link = 'https://newsdata.io/api/1/latest?apikey=pub_480828ed4d09aafac2c6eb519f41a0715d7c8&q=USA&language=en'
res = requests.get(link)
out = predict(res,gist_content)
n=1
"""
Alan Turing was a British mathematician and computer scientist who made important contributions to the fields of mathematics, cryptography, and computer science. He is widely regarded as the father of computer science and artificial intelligence.
"""