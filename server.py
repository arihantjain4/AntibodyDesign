import random

from fastapi import FastAPI

possible_aas = list("ACDEFGHIKLMNPQRSTVWY")

app = FastAPI()


def combine(seq):
    x = ""
    for i in seq:
        x += i
    return x

@app.get("/")
def read_root():
    return "invalid"

@app.get("/input")
def read_item(seq: str = None):
    # convert seq to array
    seq = list(seq)
    current_index = 0
    for change in range(10):
        current_index += random.randint(0, 20)
        num_to_change = random.randint(5, 16)
        for i in range(num_to_change):
            if current_index >= len(seq):
                break
            seq[current_index] = possible_aas[random.randint(0, 19)]
            current_index += 1
    return combine(seq)
