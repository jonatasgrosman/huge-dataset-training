import gc
import torch
import datasets
import random
from transformers import AutoTokenizer  # Or BertTokenizer
from functools import wraps
from time import time

def time_it(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time()))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time())) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} s")
    return _time_it


"""
Toda a análise foi feita com base no dataset BrWaC (17GB)

Vou tentar responder essas questões:

1 - Quanto tempo leva pra criar o cache inicial do dataset e qual o tamanho dele?
    Levou 1h consumindo em média 250MB de memória e pico de 500MB, o tamanho final dele ficou em 18GB

2 - Sem o arquivo original, só com o cache, é possível usar o dataset numa boa?
    Sim, depois de processado o arquivo original pode ser apagado que ele reaproveita do cache

3 - Quanto tempo leva pra tokenizar e salvar essa versão do dataset e qual o tamanho dele?
    Levou cerca de 3h consumindo até 7.5GB de memória. Teve um custo de armazenamento adicional de cerca de 65GB para o cache intermediário. 
    Porém a versão final dos arquivos do dataset consume apenas 35GB

4 - Uma vez salvo, quanto tempo leva pra carregar a versão processada do dataset?
    Depois de salvo levou apenas 8s pra carregar o dataset e consumiu cerca de 180MB

5 - Quanto de memória é consumida quando são criados batches com tamanho fixo mas com amostras randômicas utilizando o cache? A ideia aqui é simular como o cache vai se comportar durante o treinamento.
    Eu simulei 1K steps com um batch de 2K com seleção aleatória. O consumo máximo de memória foi de 330MB, esse processo levou 25min pra ser conncluído

6 - Como seria o consumo de memória e tempo se ao invés de fazer o load dessa versão já tokenizada eu fizesse a tokenização on-the-fly?
    A julgar pelo tempo que levou o processo de tokenização da questão 3, nem vou testar essa hipotese

7 - Qual a diferença de tempo em usar o datasets.select por item X por batch?
    por item = 25min (5_mprofile_20210113132657.dat)
    por batch = 11s usando a msm qtd de memória... holy shit! (7_mprofile_20210115131308.dat)

    Acho que vamos ter que implementar um dataloader que carregue os items lote 
    https://github.com/pytorch/pytorch/issues/13023
    https://github.com/pytorch/pytorch/issues/21645
    https://gist.github.com/SsnL/205a4cd2e4e631a42cc9d8e879a296dc
"""

DATA_DIR = "/Users/jonatas/data/brwac"
CACHE_DIR = "/Users/jonatas/data/brwac/cache"
PROCESSED_DATA_CACHE_DIR = "/Users/jonatas/data/brwac/processed_data"
TOKENIZER = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)

"""
BrWaC structure

[{
    "doc_id": "netg-1afc73",
    "title": "ESPUMA MARROM CHAMADA ‟NINGUÃÂM MERECE‟ - paulo soavinski",
    "uri": "http://blogoosfero.cc/ilhadomel/pousadasilhadomel.com.br/espuma-marrom-chamada-ninguem-merece"
    "paragraphs": {
        "sentences": [
            ["text", "text"], ...
        ]
    }
}, ...]

"""

def load():
    brwac_dataset = datasets.load_dataset("brwac", split='train', data_dir=DATA_DIR, cache_dir=CACHE_DIR)
    return brwac_dataset


def encode(doc):

    sentences = []

    # TODO: fazer um PR pro HG pra transformar o paragraphs->sentences em body->paragraphs
    for paragraph in doc.get("paragraphs").get("sentences"):
        sentences = sentences + paragraph

    encoded_sentences = TOKENIZER(sentences).get("input_ids")

    return {"encoded_sentences": encoded_sentences}


def preprocess_data():
    
    brwac_dataset = load()
    # brwac_dataset = brwac_dataset.select(range(100)) # selecting a tiny subset for test

    print("original format 1st doc")
    print(brwac_dataset[0])
    
    # encoding and removing all the other dataset columns
    processed_dataset = brwac_dataset.map(lambda doc: encode(doc), remove_columns=["paragraphs", "title", "doc_id", "uri"])
    print("processed format 1st doc")
    print(processed_dataset[0])

    # removing one-line documents
    print(f"dataset size pre one-line filter: {len(processed_dataset)}")
    processed_dataset = processed_dataset.filter(lambda doc: len(doc.get("encoded_sentences")) > 1)
    print(f"dataset size pos one-line filter: {len(processed_dataset)}")

    print("saving data...")
    processed_dataset.save_to_disk(PROCESSED_DATA_CACHE_DIR)


def load_preprocessed_data(print_first_sample=True):

    reloaded_dataset = datasets.load_from_disk(PROCESSED_DATA_CACHE_DIR)

    if print_first_sample:

        for encoded_sentence in reloaded_dataset[0].get("encoded_sentences"):
            print(f"{'=' * 100}")
            print(f"ids: {encoded_sentence}")
            print(f"tokens: {TOKENIZER.convert_ids_to_tokens(encoded_sentence)}")
            print(f"decoded: {TOKENIZER.decode(encoded_sentence)}")

    return reloaded_dataset

def random_batch_iteration(batch_size=2000, max_steps=1000, select_in_batch=False):

    reloaded_dataset = load_preprocessed_data(False)

    dataset_indexes = list(range(len(reloaded_dataset)))
    random.shuffle(dataset_indexes) # shuffle indexes

    batch = []
    step = 1
    for i in dataset_indexes:

        # reloaded_dataset[i] increases memory indefinitely, we'll need to use reloaded_dataset.select to prevent OOM
        if select_in_batch:
            batch.append(i)
        else:
            batch.append(reloaded_dataset.select([i])) # append sample to a list for naive batch loading emulation

        if len(batch) % batch_size == 0: # emulate an optimization step

            if select_in_batch:
                batch = reloaded_dataset.select(batch) # using ids for naive batch loading emulation

            print("Use your imagination... pretend I'm using the batch for some optimization process")
            print(f"step {step}/{max_steps}")
            step += 1

            # emulate batch reconstruction (trying to force memory deallocation)
            # del batch
            # gc.collect()
            batch = []
        
        if step == max_steps:
            break

@time_it
def main():
    # load()
    # preprocess_data()
    # load_preprocessed_data()
    random_batch_iteration(select_in_batch=False)

if __name__ == "__main__":
    main()