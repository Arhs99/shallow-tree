import time
from shallowtree.interfaces.full_tree_search import Expander
import multiprocessing as mp

config_filename = '/home/kostas/data/aizynth/config_remote.yml'

def worker(sub_smiles):
    pass

if __name__ == '__main__':
    smiles = [
        "CN(C)CC(O)COc1ccc(Nc2nccc(N(Cc3ccccc3)c3cc(Cl)ccc3Cl)n2)cc1",
        "O=C(Nc1n[nH]c2cc(-c3ccccc3)ccc12)C1CC1",
        "O=C(Nc1n[nH]c2nc(-c3ccc(O)c(Br)c3)ccc12)C1CC1",
        "O=C(Nc1ccc(N2CCNCC2)cc1)c1n[nH]cc1Nc1cc(Oc2ccccc2)ncn1",
        "CN(C)CC(O)COc1ccc(Nc2cc(Nc3ccccc3F)ncn2)cc1",
        "CC(C)n1cnc2c(N)nc(NCCCO)nc21",
        "Cc1ccc2[nH]cc(C(=O)C(=O)N3CCc4ccccc43)c2c1",
        "CCSc1nn2c(Cc3ccccc3)nnc2s1",
        "CCc1cc2c(N/N=C\c3cccs3)nc(-c3ccccc3)nc2s1",
        "NCC(=O)NS(=O)(=O)c1ccc(Nc2nc(N)n(C(=O)c3c(F)cccc3F)n2)cc1",
        "CC1(C)OCC(=O)Nc2cc(Nc3nc(NCC(F)(F)F)c4occc4n3)ccc21",
        "NS(=O)(=O)c1ccc(N/N=C2\C(=O)Nc3cc(Br)ccc32)cc1",
        "O=C1Nc2ccc(F)c3c(OCCCO)cc(-c4ccc[nH]4)c1c23",
        "COc1cc(C)c(Sc2cnc(NC(=O)c3ccc(CNC(C)C(C)(C)C)cc3)s2)cc1C(=O)N1CCN(C(C)=O)CC1",
        "Cc1ccc(Nc2cc(Nc3ccc(OCC(O)CN(C)C)cc3)ncn2)cc1",
        "CC(C)(C)c1ccc(Nc2nccc(-c3cnn4ncccc34)n2)cc1",
        "COc1cc(O)c2c(c1)C(=O)c1cc(C)c(O)c(O)c1C2=O",
        "CC(C)n1cnc2c(NC3CCCC3)nc(NCCCO)nc21",
        "O=C(Nc1n[nH]c2cc(-c3ccc(F)cc3)ccc12)C1CC1",
        "O=C(Nc1n[nH]c2nc(-c3ccco3)c(Br)cc12)C1CCN(Cc2ccccc2)C1",
        "CC(=O)Nc1c[nH]nc1-c1nc2ccccc2[nH]1",
        "O=C(Nc1c[nH]nc1-c1nc2ccc(CN3CCOCC3)cc2[nH]1)c1ccc(F)cc1",
        "O=C(Nc1c[nH]nc1-c1nc2cc(CN3CCOCC3)ccc2[nH]1)Nc1c(F)cccc1F",
        "CC1NCCCC1C(=O)Nc1ncc(SCc2ncc(C(C)(C)C)o2)s1",
        "O=C(Nc1n[nH]c2cc(-c3cccs3)ccc12)C1CC1",
        "CC(C)(C)c1cnc(CSc2cnc(NCC3CCNCC3)s2)o1",
        "Cc1ccc(C(=O)NC2CC2)cc1-c1ccc2c(C3CCNCC3)noc2c1",
        "CN(C)c1cc2c(Nc3ccc4c(cnn4Cc4ccccc4)c3)ncnc2cn1",
        "Nc1ccccc1NC(=O)c1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
        "N#Cc1c(NC(=O)c2cccc3ccccc23)sc2c1CCCC2",
        "NS(=O)(=O)c1ccc(N/N=C2\C(=O)Nc3ccc4ncsc4c32)cc1",
        "NS(=O)(=O)c1ccc(N/N=C2\C(=O)Nc3cc(Oc4ccccc4)ccc32)cc1",
        "CNC(=O)c1nn(C)c2c1C(C)(C)Cc1cnc(Nc3ccc(CN4CCN(C)CC4)cc3)nc1-2",
        "CC(C)(C)c1cc2c(N/N=C\c3cccc(CN)n3)ncnc2s1",
        "COc1cccc2c1c(Cl)c1c3c(cc(O)c(O)c32)C(=O)N1",
    ]

    start = time.time()
    expander = Expander(configfile=config_filename)
    expander.expansion_policy.select_first()
    print(f'Load exp policy: {time.time() - start} sec')
    expander.filter_policy.select_first()
    print(f'Load filter policy: {time.time() - start} sec')
    expander.stock.select_first()
    print(f'Load stock: {time.time() - start} sec')

    # df = expander.context_search(
    #     ['c4ccc(c3ccc2c(C1CCOC1)c[nH]c2c3)cc4'], scaffold_str='[*]c1c[nH]c2cc(-c3ccccc3)ccc12'
    # )

    df = expander.search_tree(
        ['CC(c2c[nH]c3cc(c1ccccc1)ccc23)C5CC(OCc4ccccc4Cl)C5']
        # smiles[:5]
        # ['C(Nc1n[nH]c2cc(-c3ccccc3)ccc12)C1CC1']
        # 'Cc1cccc(c1N(CC(=O)Nc2ccc(cc2)c3ncon3)C(=O)C4CCS(=O)(=O)CC4)C'
    )
    print(df.to_csv())
