import csv
import os
import xml.etree.ElementTree as ET

salsa_xmlfile = os.path.join(os.path.dirname(__file__), "salsa_release.xml")
salsa_splitfile = os.path.join(os.path.dirname(__file__), "salsa_splits.txt")
salsa_csvfilename = os.path.join(os.path.dirname(__file__), 'salsa.csv')

salsa_csvfile = open(salsa_csvfilename, 'w', newline='')
fieldnames = ["DATA_SOURCE", "FRAME_ID", "TOKENIZED_SENTENCE", "GLOBAL_SENTENCE_ID", "LU_INDEX", "LU_INDEX_PART", "LU", "LU_LEMMA", "LU_LEMMA_PART", "LU_LEMMA_FULL", "SUBSTITUTES", "split"]

## get splits
splits = {}
with open(salsa_splitfile) as splitfile:
    for line in splitfile:
        splitpart, ids = line.split(':', 2)
        ids = set(ids.strip().split(' '))
        splits[splitpart] = ids

assert splits['Training + Development'] == (splits['Training'].union(splits['Development']))
del splits['Training + Development']
print('Read splits.')

tree = ET.parse(salsa_xmlfile)
root = tree.getroot()

writer = csv.DictWriter(salsa_csvfile, fieldnames=fieldnames, dialect='excel-tab')
writer.writeheader()

def get_target_nodes(target_node_ids, sentence):

    ## using a set - nodes can appear twice
    target_nodes = set()

    for node_id in target_node_ids:
        node = sentence.find('graph/terminals/t[@id=\'' + node_id + '\']')
        if node is not None:
            target_nodes.add(node)
        else:
            ## node is not a terminal node, get corresponding nodes
            new_target_nodes = [tnode.attrib['idref'] for tnode in sentence.findall('graph/nonterminals/nt[@id=\'' + node_id + '\']/edge')]
            if not new_target_nodes:
                ## node is neither a nonterminal node but a split word
                ## TODO what to do?
                pass
            else:
                target_nodes.update(get_target_nodes(new_target_nodes, sentence))

    return target_nodes



for sentence in root.iter('s'):

    sentence_info = {'DATA_SOURCE': 'SALSA2'}
    sentence_info['TOKENIZED_SENTENCE'] = " ".join([t.attrib['word'] for t in sentence.iter('t')])
    sentence_info['GLOBAL_SENTENCE_ID'] = "salsa2_" + sentence.attrib['id']

    is_train = False
    is_devel = False
    is_test  = False

    if sentence.attrib['id'] in splits['Training']:
        is_train = True
    if sentence.attrib['id'] in splits['Development']:
        is_devel = True
    if sentence.attrib['id'] in splits['Test']:
        is_test = True

    ## Sentence is not part of any split - skip
    if sum([is_train, is_devel, is_test]) == 0:
        continue

    assert sum([is_train, is_devel, is_test]) == 1

    if is_train:
        sentence_info['split'] = 'train'
    elif is_devel:
        sentence_info['split'] = 'dev'
    elif is_test:
        sentence_info['split'] = 'test'

    for frame in sentence.iter('frame'):

        ## get target nodes

        targets = frame.findall('target')
        ## make sure that there is only one target
        assert len(targets) == 1
        target = targets[0]

        ## get all terminal nodes for the target
        target_node_ids = [fenode.attrib['idref'] for fenode in target.findall('fenode')]
        target_nodes = get_target_nodes(target_node_ids, sentence)

        ## only use verbal targets that correspond to lemma or headlemma
        lu_lemma = target.attrib.get('headlemma', target.attrib['lemma'])

        verb_target = [target_node for target_node in target_nodes if target_node.attrib['pos'].startswith('V')]
        ptkvz_target = [target_node for target_node in target_nodes if target_node.attrib['pos'] == 'PTKVZ']

        if verb_target:

            main_target = None
            part_target = []

            if ptkvz_target:
                if len(ptkvz_target) == 1 and len(verb_target) == 1:
                    main_target = verb_target[0]
                    part_target = ptkvz_target

                    ## fix annotation error in s723
                    if lu_lemma == "anführen" and main_target.attrib['lemma'] == "fahren":
                        main_target.attrib['lemma'] = "führen"
                    ## fix annotaiton error in bleiben_s3099_f1
                    if lu_lemma == "bleiben" and main_target.attrib['lemma'] == "bleiben":
                        lu_lemma = "übrigbleiben"
                    ## fix annotation error in s6410
                    if lu_lemma == "mitteilen" and main_target.attrib['lemma'] == "mitteilen":
                        main_target.attrib['lemma'] = "teilen"
                    ## fix annotation error in s9200
                    if lu_lemma == "zurückziehen" and main_target.attrib['lemma'] == "zeihen":
                        main_target.attrib['lemma'] = "ziehen"
                    ## fix annotation error in hetzen_s17899_f0
                    if lu_lemma == "hetzen" and main_target.attrib['lemma'] == "hetzen":
                        lu_lemma = "hinterherhetzen"
                    ## fix annotation error in Auge_s25972_f1
                    if lu_lemma == "Auge" and main_target.attrib['lemma'] == "drücken":
                        lu_lemma = "zudrücken"
                    ## fix annotation error in s34060
                    if lu_lemma == "ansehen" and main_target.attrib['lemma'] == "sähen":
                        main_target.attrib['lemma'] = "sehen"
                    ## fix annotation error in reden_s34870_f2
                    if lu_lemma == "reden" and main_target.attrib['lemma'] == "reden":
                        lu_lemma = "herausreden"

                    assert lu_lemma == part_target[0].attrib['lemma'] + main_target.attrib['lemma']
                else:
                    ## "lassen übrig wünschen zu"oder "ab auf der wir vermeintlich sitzen sägen der Ast"
                    ## nur das Hauptverb als Target nehmen

                    main_verb_target = [target_node for target_node in verb_target if target_node.attrib['lemma'] == lu_lemma]
                    assert len(main_verb_target) == 1
                    main_target = main_verb_target[0]

            elif len(verb_target) > 1:
                ## "denken geben" oder "müssen rechnen"
                ## nur das Hauptverb als Target nehmen

                main_verb_target = [target_node for target_node in verb_target if target_node.attrib['lemma'] == lu_lemma]
                assert len(main_verb_target) == 1
                main_target = main_verb_target[0]
            else:
                main_target = verb_target[0]

            frame_info = {}
            frame_info['FRAME_ID'] = frame.attrib['name']
            frame_info['LU_INDEX'] = int(main_target.attrib['id'].split('_')[-1]) - 1
            frame_info['LU_INDEX_PART'] = ','.join([str(int(t.attrib['id'].split('_')[-1]) - 1) for t in part_target])
            frame_info['LU'] = main_target.attrib['word']
            frame_info['LU_LEMMA'] = main_target.attrib['lemma']
            frame_info['LU_LEMMA_PART'] = ",".join([t.attrib['lemma'] for t in part_target])
            frame_info['LU_LEMMA_FULL'] = lu_lemma

            writer.writerow({**sentence_info, **frame_info})

salsa_csvfile.close()
