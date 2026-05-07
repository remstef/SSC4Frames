# coding=utf-8
# copied and adapted from Swabha Swayamdipta
# https://github.com/swabhs/open-sesame/blob/master/sesame/preprocess.py

from __future__ import division

'''
Reads XML files containing FrameNet 1.$VERSION annotations, and converts them to a CoNLL 2009-like format.
'''
import codecs
import os.path
import sys
import tqdm


import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


import xml.etree.ElementTree as et
from optparse import OptionParser

VERSION="1.7"
DATA_DIR="./"
PARSER_DATA_DIR = DATA_DIR + "open_sesame_v1_data/fn" + VERSION + "/"
TRAIN_EXEMPLAR = PARSER_DATA_DIR + "fn" + VERSION + ".exemplar.train.syntaxnet.conll"
TRAIN_FTE = PARSER_DATA_DIR + "fn" + VERSION + ".fulltext.train.syntaxnet.conll"
DEV_CONLL = PARSER_DATA_DIR + "fn" + VERSION + ".dev.syntaxnet.conll"
TEST_CONLL = PARSER_DATA_DIR + "fn" + VERSION + ".test.syntaxnet.conll"
FN_DATA_DIR = DATA_DIR + "fndata-" + VERSION + "/"
LU_INDEX = FN_DATA_DIR + "luIndex.xml"
FULLTEXT_DIR = FN_DATA_DIR + "fulltext/"
LU_DIR = FN_DATA_DIR + "lu/"
FULLTEXT_DIR = FN_DATA_DIR + "fulltext/"
FRAME_DIR = FN_DATA_DIR + "frame/"
FRAME_REL_FILE = FN_DATA_DIR + "frRelation.xml"

TEST_FILES = [
        "ANC__110CYL067.xml",
        "ANC__110CYL069.xml",
        "ANC__112C-L013.xml",
        "ANC__IntroHongKong.xml",
        "ANC__StephanopoulosCrimes.xml",
        "ANC__WhereToHongKong.xml",
        "KBEval__atm.xml",
        "KBEval__Brandeis.xml",
        "KBEval__cycorp.xml",
        "KBEval__parc.xml",
        "KBEval__Stanford.xml",
        "KBEval__utd-icsi.xml",
        "LUCorpus-v0.3__20000410_nyt-NEW.xml",
        "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
        "LUCorpus-v0.3__enron-thread-159550.xml",
        "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
        "LUCorpus-v0.3__SNO-525.xml",
        "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
        "Miscellaneous__Hound-Ch14.xml",
        "Miscellaneous__SadatAssassination.xml",
        "NTI__NorthKorea_Introduction.xml",
        "NTI__Syria_NuclearOverview.xml",
        "PropBank__AetnaLifeAndCasualty.xml",
        ]

DEV_FILES = [
        "ANC__110CYL072.xml",
        "KBEval__MIT.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
        "Miscellaneous__Hijack.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]

EMPTY_LABEL = "_"

lemmatizer = nltk.stem.WordNetLemmatizer()

class SentenceAnnotation(object):

    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.postags = []
        self.nltkpostags = []
        self.nltklemmas = []
        self.foundpos = False # either BNC or PENN annotations
        self.stindices = {}
        self.enindices = {}

    def add_token(self, startend):
        st, en = startend
        st = int(st)
        en = int(en)
        self.stindices[st] = len(self.tokens)
        self.enindices[en] = len(self.tokens)

    def normalize_tokens(self, logger):
        if len(self.stindices) != len(self.enindices):
            logger.write("\t\tIssue: overlapping tokenization for multiple tokens\n")
            return
        start = {}
        idx = 0
        for s in sorted(self.stindices):
            self.stindices[s] = idx
            start[idx] = s
            idx += 1
        end = {}
        idx = 0
        for t in sorted(self.enindices):
            self.enindices[t] = idx
            end[idx] = t
            if idx > 0 and end[idx - 1] > start[idx]:
                logger.write("\t\tIssue: overlapping tokenization of neighboring tokens\n")
                return
            token = self.text[start[idx] : t + 1].strip()
            if " " in token:
                logger.write("\t\tIssue: incorrect tokenization "  + token + "\n")
                return
            if token == "": continue
            self.tokens.append(token)
            idx += 1
        try:
            self.nltkpostags = [ele[1] for ele in nltk.pos_tag(self.tokens)]
            for idx in range(len(self.tokens)):
                tok = self.tokens[idx]
                if self.nltkpostags[idx].startswith("V"):
                    self.nltklemmas.append(lemmatizer.lemmatize(tok, pos='v'))
                else:
                    self.nltklemmas.append(lemmatizer.lemmatize(tok))
        except IndexError:
            print(self.tokens)
            print(nltk.pos_tag(self.tokens))
        return True

    def get_tokens_by_offset(self, startend):
        st, en = startend
        st = int(st)
        en = int(en)
        if st not in self.stindices or en not in self.enindices:
            raise Exception("\t\tBug: broken tokenization", st, en)
        return self.stindices[st], self.enindices[en]

    def add_postag(self, postag):
        self.foundpos = True
        self.postags.append(postag)

    def size(self):
        return len(self.tokens)

    def info_at_idx(self, idx):
        if len(self.tokens) <= idx :
            raise Exception("\t\tBug: invalid index", idx)
        if len(self.postags) <= idx:
            postag = EMPTY_LABEL
        else:
            postag = self.postags[idx]
        return self.tokens[idx], postag, self.nltkpostags[idx], self.nltklemmas[idx]


class FrameAnnotation(object):

    def __init__(self, lu, frame, sent, frameid='', luid=''):
        self.lu = lu
        self.frame = frame
        self.sent = sent
        self.target = set([])
        self.foundtarget = False
        self.fe = {}
        self.foundfes = False
        self.frameid = frameid
        self.luid = luid

    def add_fe(self, offset, arglabel, logger):
        try:
            st, en = self.sent.get_tokens_by_offset(offset)
        except Exception:
            logger.write("\t\tIssue: broken tokenization for FE\n")
            return
        self.foundfes = True
        for idx in range(st, en + 1):
            if idx in self.fe:
                raise Exception("\t\tIssue: duplicate FE at ", idx, self.fe)

        # BIOS tagging
        if st == en:
            self.fe[st] = "S-" + arglabel
        else:
            self.fe[st] = "B-" + arglabel
            for idx in range(st+1, en+1):
                if idx in self.fe:
                    raise Exception("duplicate FE at ", idx, offset, arglabel)
                self.fe[idx] = "I-" + arglabel


    def add_target(self, offset, logger):
        try:
            st, en = self.sent.get_tokens_by_offset(offset)
        except Exception:
            logger.write("\t\tIssue: broken tokenization for target\n")
            return
        self.foundtarget = True
        for idx in range(st, en + 1):
            if idx in self.target:
                logger.write("\t\tIssue: duplicate target at " + str(idx) + "\n")
            self.target.add(idx)

    def info_at_idx(self, idx):
        token, postag, nltkpostag, nltklemma = self.sent.info_at_idx(idx)
        lexunit = frm = "_"
        role = "O"

        if idx in self.target:
            lexunit = self.lu
            frm = self.frame

        if idx in self.fe:
            role = self.fe[idx]

        return token, postag, nltkpostag, nltklemma, lexunit, frm, role

    def __hash__(self):
        return hash((self.lu, self.frame, frozenset(self.target)))

    def __eq__(self, other):
        return self.lu == other.lu and self.frame == other.frame and self.target == other.target

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # true at the same time
        return not(self == other)



optpr = OptionParser()
optpr.add_option("--exemplar", action="store_true", default=False)
(options, args) = optpr.parse_args()

logger = open("preprocess-fn{}.log".format(VERSION), "w")

trainf = TRAIN_EXEMPLAR
ftetrainf = TRAIN_FTE
devf = DEV_CONLL
testf = TEST_CONLL

trainsentf = TRAIN_EXEMPLAR + ".sents"
ftetrainsentf = TRAIN_FTE + ".sents"
devsentf = DEV_CONLL + ".sents"
testsentf = TEST_CONLL + ".sents"

relevantfelayers = ["Target", "FE"]
relevantposlayers = ["BNC", "PENN"]
ns = {'fn' : 'http://framenet.icsi.berkeley.edu'}

firsts = {trainf: True,
          devf: True,
          testf: True,
          ftetrainf: True}
sizes = {trainf: 0,
         devf: 0,
         testf: 0,
         ftetrainf: 0}
totsents = numsentsreused = fspno = numlus = 0.0
isfirst = isfirstsent = True


def write_to_conll(outf, fsp, firstex, sentid, sentenceplain='', framenet_sentenceid='-1', source=''):
    mode = "a"
    if firstex:
        mode = "w"

    with codecs.open(outf, mode, "utf-8") as outf:
        outf.write(f'# SOURCE={source} \n')  # Source File
        outf.write(f'# SID={framenet_sentenceid} \n')  # SID
        outf.write(f'# TEXT={sentenceplain} \n')  # plain sentence
        outf.write(f'# LU={fsp.lu} \n')  # LU (Target)
        outf.write(f'# LUID={fsp.luid} \n')  # LUID (Target)
        outf.write(f'# FRAMENAME={fsp.frame} \n')  # Frame Name
        outf.write(f'# FRAMEID={fsp.frameid} \n')  # Frame ID

        for i in range(fsp.sent.size()):
            token, postag, nltkpostag, nltklemma, lu, frm, role = fsp.info_at_idx(i)

            outf.write(str(i + 1) + "\t")  # ID = 0
            outf.write(str(token.encode('utf-8')) + "\t")  # FORM = 1
            outf.write("_\t" + nltklemma + "\t")  # LEMMA PLEMMA = 2,3
            outf.write(postag + "\t" + nltkpostag + "\t")  # POS PPOS = 4,5
            outf.write(str(sentid - 1) + "\t_\t")  # FEAT PFEAT = 6,7 ~ replacing FEAT with sentence number
            outf.write("_\t_\t")  # HEAD PHEAD = 8,9
            outf.write("_\t_\t")  # DEPREL PDEPREL = 10,11
            outf.write(lu + "\t" + frm + "\t")  # FILLPRED PRED = 12,13
            outf.write(role + "\n")  #APREDS = 14

        outf.write("\n")  # end of sentence
        outf.close()


def write_to_sent_file(outsentf, sentence, isfirstsent, sentenceid='-1', framemetadata=[]):
    mode = "a"
    if isfirstsent: mode = "w"

    with codecs.open(outsentf, mode, "utf-8") as outf:
        outf.write(sentenceid + '\t' + sentence + '\t' + str(framemetadata) + "\n")  # end of sentence
        outf.close()


def process_xml_labels(label, layertype):
    try:
        st = int(label.attrib["start"])
        en = int(label.attrib["end"])
    except KeyError:
        logger.write("\t\tIssue: start and/or end labels missing in " + layertype + "\n")
        return
    return (st, en)


def process_sent(sent, outsentf, isfirstsent, sentenceid='-1', framemetadata=[]):
    senttext = ""
    for t in sent.findall('fn:text', ns):  # not a real loop
        senttext = t.text

    write_to_sent_file(outsentf, senttext, isfirstsent, sentenceid, framemetadata)
    sentann = SentenceAnnotation(senttext)

    for anno in sent.findall('fn:annotationSet', ns):
        for layer in anno.findall('fn:layer', ns):
            layertype = layer.attrib["name"]
            if layertype in relevantposlayers:
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    sentann.add_token(startend)
                    sentann.add_postag(label.attrib["name"])
                if sentann.normalize_tokens(logger) is None:
                    logger.write("\t\tSkipping: incorrect tokenization\n")
                    return None, None
                break
        if sentann.foundpos:
            break

    if not sentann.foundpos:
        # TODO do some manual tokenization
        logger.write("\t\tSkipping: missing POS tags and hence tokenization\n")
        return None, None
    return sentann, senttext


def get_all_fsps_in_sent(sent, sentann, fspno, lex_unit, frame, isfulltextann, corpus, frame_id='', lex_unit_id=''):
    numannosets = 0
    fsps = {}
    fspset = set([])

    # get all the FSP annotations for the sentece : it might have multiple targets and hence multiple FSPs
    for anno in sent.findall('fn:annotationSet', ns):
        annotation_id = anno.attrib["ID"]
        if annotation_id == "2019791" and VERSION == "1.5":
            # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
            continue
        numannosets += 1
        if numannosets == 1:
            continue
        anno_id = anno.attrib["ID"]
        if isfulltextann: # happens only for fulltext annotations
            if "luName" in anno.attrib:
                if anno.attrib["status"] == "UNANN" and "test" not in corpus: # keep the unannotated frame-elements only for test, to enable comparison
                    continue
                lex_unit = anno.attrib["luName"]
                framename = anno.attrib["frameName"]
                frameid = str(anno.attrib["frameID"])
                lex_unit_id = str(anno.attrib["luID"])
                if framename == "Test35": continue # bogus frame
            else:
                continue
            logger.write("\tannotation: " + str(anno_id) + "\t" + framename + "\t" + lex_unit + "\n")
            fsp = FrameAnnotation(lex_unit, framename, sentann, frameid, lex_unit_id)
        else:
            framename = frame
            fsp = FrameAnnotation(lex_unit, framename, sentann, frame_id, lex_unit_id)

        for layer in anno.findall('fn:layer', ns):  # not a real loop
            layertype = layer.attrib["name"]
            if layertype not in relevantfelayers:
                continue
            if layertype == "Target" :
                for label in layer.findall('fn:label', ns):  # can be a real loop
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        break
                    fsp.add_target(startend, logger)
            elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        if "itype" in label.attrib:
                            logger.write("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                            continue
                        else:
                            break
                    fsp.add_fe(startend, label.attrib["name"], logger)

        if not fsp.foundtarget:
            logger.write("\t\tSkipping: missing target\n")
            continue
        if not fsp.foundfes:
            logger.write("\t\tIssue: missing FSP annotations\n")
        if fsp not in fspset:
            fspno += 1
            fsps[anno_id] = fsp
            fspset.add(fsp)
        else:
            logger.write("\t\tRepeated frames encountered for same sentence\n")

    return numannosets, fspno, fsps


def get_annoids(filelist, outf, outsentf):
    annos = []
    isfirstex = True
    fspno = 0
    numsents = 0
    invalidsents = 0
    repeated = 0
    totfsps = 0
    sents = set([])
    isfirstsentex = True

    for tfname in tqdm.tqdm(filelist):
        tfname = os.path.join(FULLTEXT_DIR, tfname)
        logger.write("\n" + tfname + "\n")
        if not os.path.isfile(tfname):
            continue
        with codecs.open(tfname, 'rb', 'utf-8') as xml_file:
            tree = et.parse(xml_file)

        root = tree.getroot()
        for sentence in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
            numsents += 1
            sentenceid = str(sentence.attrib["ID"])
            logger.write("sentence:\t" + sentenceid + "\n")
            frameannometadata = [ ]
            for annotation in sentence.iter('{http://framenet.icsi.berkeley.edu}annotationSet'):
                annotation_id = annotation.attrib["ID"]
                if annotation_id == "2019791" and VERSION == "1.5":
                    # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
                    continue
                if "luName" in annotation.attrib and "frameName" in annotation.attrib:
                    annos.append(annotation.attrib["ID"])
                frameannometadata_i = { }
                for frameannometadata_key in ['luID', 'luName', 'frameID', 'frameName']:
                    if frameannometadata_key in annotation.attrib:
                        frameannometadata_i[frameannometadata_key] = str(annotation.attrib[frameannometadata_key])
                if len(frameannometadata_i):
                    frameannometadata.append(frameannometadata_i)
                # luID="4654" luName="December.n" frameID="229" frameName="Calendric_unit" 
            # get the tokenization and pos tags for a sentence
            sentann, senttext = process_sent(sentence, outsentf, isfirstsentex, sentenceid, frameannometadata)
            isfirstsentex = False
            if sentann is None:
                invalidsents += 1
                logger.write("\t\tIssue: Token-level annotations not found\n")
                continue

            # get all the different FSP annotations in the sentence
            x, fspno, fsps = get_all_fsps_in_sent(sent=sentence, sentann=sentann, fspno=fspno, lex_unit=None, frame=None, isfulltextann=True, corpus=outf, frame_id=None, lex_unit_id=None)
            totfsps += len(fsps)
            if len(fsps) == 0: invalidsents += 1
            if sentann.text in sents:
                repeated += 1
            for fsp in list(fsps.values()):
                sents.add(sentann.text)
                write_to_conll(outf, fsp, isfirstex, numsents, senttext, framenet_sentenceid=sentenceid, source=tfname)
                sizes[outf] += 1
                isfirstex = False
        xml_file.close()
    sys.stderr.write("# total sents processed = %d\n" % numsents)
    sys.stderr.write("# repeated sents        = %d\n" % repeated)
    sys.stderr.write("# invalid sents         = %d\n" % invalidsents)
    sys.stderr.write("# sents in set          = %d\n" % len(sents))
    sys.stderr.write("# annotations           = %d\n" % totfsps)
    return annos


def process_fulltext():
    sys.stderr.write("\nReading {} fulltext data ...\n".format(VERSION))

    # read and write all the test examples in conll
    logger.write("\n\nTEST\n\n")
    sys.stderr.write("TEST\n")
    test_annos = get_annoids(TEST_FILES, testf, testsentf)

    # read and write all the dev examples in conll
    logger.write("\n\nDEV\n\n")
    sys.stderr.write("DEV\n")
    dev_annos = get_annoids(DEV_FILES, devf, devsentf)

    # read all the full-text train examples in conll
    train_fte_files = []
    for f in os.listdir(FULLTEXT_DIR):
        if f not in TEST_FILES and f not in DEV_FILES and not f.endswith("xsl"):
            train_fte_files.append(f)
    logger.write("\n\nFULLTEXT TRAIN\n\n")
    sys.stderr.write("FULLTEXT TRAIN\n")
    get_annoids(train_fte_files, ftetrainf, ftetrainsentf)

    return dev_annos, test_annos


def process_lu_xml(lufname, all_exemplars, dev_annos, test_annos):
    global totsents, numsentsreused, fspno, numlus, isfirst, isfirstsent
    with codecs.open(lufname, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()
    ns = {'fn': 'http://framenet.icsi.berkeley.edu'}

    frame = root.attrib["frame"]
    frame_id = root.attrib["frameID"]
    lex_unit = root.attrib["name"]
    lex_unit_id = root.attrib["ID"]
    frameannometadata = [ {
        'luID' : lex_unit_id, 
        'luName' : lex_unit,
        'frameID' : frame_id,
        'frameName' : frame,
        'source' : lufname,
     } ]
    logger.write("\n" + lufname + "\t" + frame + "\t" + lex_unit + "\n")

    sentno = 0
    for sent in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
        sentno += 1
        # get the tokenization and pos tags for a sentence
        sent_id = int(sent.attrib["ID"])
        logger.write("sentence:\t" + str(sent_id) + "\n")

        sentann, senttext = process_sent(sent, trainsentf, isfirstsent, str(sent_id), frameannometadata)
        isfirstsent = False

        if sentann is None:
            logger.write("\t\tIssue: Token-level annotations not found\n")
            continue

        # get all the different FSP annotations in the sentence
        numannosets, fspno, fsps = get_all_fsps_in_sent(sent=sent, sentann=sentann, fspno=fspno, lex_unit=lex_unit, frame=frame, isfulltextann=False, corpus="exemplartrain", frame_id=frame_id, lex_unit_id=lex_unit_id)
        for anno_id in fsps:
            if anno_id in test_annos or anno_id in dev_annos:
                continue
            else:
                if sent_id in all_exemplars:
                    all_exemplars[sent_id].append((fsps[anno_id], frameannometadata))
                else:
                    all_exemplars[sent_id] = [(fsps[anno_id], frameannometadata)]
                sizes[trainf] += 1

        if numannosets > 2:
            numsentsreused += (numannosets - 2)
    numlus += 1
    xml_file.close()

    logger.write(lufname + ": total sents = " + str(sentno) + "\n")
    totsents += sentno

    return all_exemplars

def process_exemplars(dev_annos, test_annos):
    global totsents, numsentsreused, fspno, numlus, isfirst
    # get the names of all LU xml files
    all_lus = []
    for f in os.listdir(LU_DIR):
        luf = os.path.join(LU_DIR, f)
        if luf.endswith("xsl"):
            continue
        all_lus.append(luf)
    sys.stderr.write("\nReading exemplar data from " + str(len(all_lus)) + " LU files...\n")

    logger.write("\n\nTRAIN EXEMPLAR\n\n")

    all_exemplars = {}
    for luname in tqdm.tqdm(sorted(all_lus)):
        if not os.path.isfile(luname):
            logger.write("\t\tIssue: Couldn't find %s - strange, terminating!\n" % (luname))
            break
        all_exemplars = process_lu_xml(luname, all_exemplars, dev_annos, test_annos)

    total_exemplars = sum([len(x) for x in all_exemplars.values()])
    sys.stderr.write("\nWriting %d exemplars to %s ...\n" % (total_exemplars, trainf))
    isfirst = True
    for write_id, sentid in enumerate(sorted(all_exemplars), 1):
        for (fsp_, frameannometadata) in all_exemplars[sentid]:
            source = ';'.join([m['source'] for m in frameannometadata])
            write_to_conll(trainf, fsp_, isfirst, sentid=write_id, sentenceplain=fsp_.sent.text, framenet_sentenceid=sentid, source=source)
            isfirst = False

    sys.stderr.write("\n\n# total LU sents = %d \n" % (totsents))
    sys.stderr.write("# total LU FSPs = %d \n"  % (fspno))
    sys.stderr.write("# total LU files = %d \n" % (numlus))
    sys.stderr.write("average # FSPs per LU = %.3f \n" % (fspno / numlus))
    sys.stderr.write("# LU sents reused for multiple annotations = %d \n" % (numsentsreused))
    sys.stderr.write("\noutput file sizes:\n")
    for s in sizes:
        sys.stderr.write("%s :\t %d \n" % (s, sizes[s]))
    sys.stderr.write("\n")




if __name__ == "__main__":
    if not os.path.exists(PARSER_DATA_DIR):
        os.makedirs(PARSER_DATA_DIR)

    dev, test = process_fulltext()

    if options.exemplar:
        process_exemplars(dev, test)

    logger.close()
