import requests
import pandas as pd

protein_db = pd.DataFrame([["", "", "", "", "", ""]], columns=["UniProtKB ID", "UniProtKB Name", "Organism", "Binding Affinity", "Binding Sites", "Amino Acid Sequence"])


def get_protein(uniprotkb_id, binding_affinity):
    uniprotkb_url = "https://rest.uniprot.org/uniprotkb/"
    req = requests.get(uniprotkb_url + uniprotkb_id)
    # print(req.text)
    global protein_db
    features_str = ""
    # check if req.json() dict has key 'features'
    if 'features' not in req.json():
        return
    for feature in req.json()['features']:
        if feature['type'] == "Binding site":
            features_str += (str(feature['location']['start']['value']) + "," + str(feature['location']['end']['value'])) + ';'
    protein_db = pd.concat([protein_db, pd.DataFrame([[uniprotkb_id, req.json()['uniProtkbId'], req.json()['organism']['scientificName'], binding_affinity, features_str, req.json()['sequence']['value']]], columns=["UniProtKB ID", "UniProtKB Name", "Organism", "Binding Affinity", "Binding Sites", "Amino Acid Sequence"])])


def read_pdbbind_complexes(file_name):
    return pd.read_excel(file_name, sheet_name="Search Results")


def run():
    global protein_db
    complexes_db = read_pdbbind_complexes("500-Protein-Ligand-Complexes.xlsx")
    for i in range(len(complexes_db)):
        if type(complexes_db.iloc[i]['UniProt AC']) != str:
            continue
        spl = complexes_db.iloc[i]['UniProt AC'].split(' ')
        if not spl[0]:
            continue
        get_protein(spl[0], binding_affinity=complexes_db.iloc[i]['Affinity Data'])
    protein_db.to_csv("protein_db.csv")


if __name__ == "__main__":
    run()
