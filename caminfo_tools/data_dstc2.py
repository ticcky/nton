#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0111

import os
import argparse
from collections import defaultdict
import json
import os.path
import sys

from collections import namedtuple


ASRHyp = namedtuple('ASRHyp', ['hyp', 'score'])
ASRWCNHyp = namedtuple('ASRWCNHyp', ['hyps', 'scores'])
SLUHyp = namedtuple('SLUHyp', ['acts', 'score'])
DialogAct = namedtuple('DialogAct', ['act', 'slots'])
Slot = namedtuple('Slot', ['name', 'value'])

NEAR_INF = sys.float_info.max


class Dialog(object):
    """Dialog log.

    Representation of one dialog.

    Attributes:
        turns: A list of dialog turns.
        session_id: ID of the dialog.
    """

    def __init__(self, log, labels):
        """Initialises a dialogue object from the external format.

        Keyword arguments:
            log: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """
        self.turns = []
        self.session_id = log['session-id']

        if labels:
            for turn_json, turn_label in zip(log['turns'], labels['turns']):
                self.turns.append(Turn(turn_json, turn_label))
        else:
            for turn_json in log['turns']:
                self.turns.append(Turn(turn_json, None))

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Dialog:\n'
        repr_str += (indent + step) * ' ' + 'id: "%s",\n' % self.session_id
        repr_str += (indent + step) * ' ' + 'turns:\n'
        for turn in self.turns:
            repr_str += turn.pretty_print(indent + 2 * step, step) + '\n'
        return repr_str

    def __str__(self):
        return self.pretty_print()

    def __repr__(self):
        return 'Dialog(id="%s")' % self.session_id


class Turn(object):
    """One turn of a dialog.

    Representation of one turn in a dialog. Contains information about
    things the user asked as well as the reply from dialog manager.

    Attributes:
        turn_index: Index of the turn in the dialog.
        transcription: Correct transcription of input.
        input: Input from the user.
        ouput: Output from the dialog manager.
        restart: Whether the dialog manager decided to restart the dialog.
    """

    def __init__(self, turn, labels):
        """Initialises a turn object from the external format.

        Keyword arguments:
            log: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """
        mact = []
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        this_slot = None
        for act in mact :
            if act["act"] in ["request"]:
                this_slot = act["slots"][0][1]
            elif act["act"] in ["expl-conf", "select"]:
                this_slot = act["slots"][0][0]
        # return a dict of informable slots to mentioned values in a turn
        out = defaultdict(set)
        for act in mact :
            if "conf" in act["act"]  :
                for slot, value in act["slots"] :
                    out[slot].add(value)


        for slu_hyp in turn["input"]["live"]["slu-hyps"] :
            for act in slu_hyp["slu-hyp"] :
                for slot, value in act["slots"] :
                    if slot == "this" :
                        slot = this_slot

                    out[slot].add(value)

        mact = turn["output"]["dialog-acts"]
        for slu_hyp in turn["input"]["live"]["slu-hyps"] :
            user_act = slu_hyp["slu-hyp"]

            method="none"
            act_types = [act["act"] for act in user_act]
            mact_types = [act["act"] for act in mact]
            if "reqalts" in act_types :
                method = "byalternatives"
            elif "bye" in act_types :
                method = "finished"
            elif "inform" in act_types:
                method = "byconstraints"
                for act in [uact for uact in user_act if uact["act"] == "inform"] :
                    slots = [slot for slot, _ in act["slots"]]
                    if "name" in slots :
                        method = "byname"

            if method != 'none':
                out['method'] = method

        if 'slot' in out:
            for val in out['slot']:
                out['req_%s' % val] = True
            del out['slot']

        self.slots_mentioned = out.keys()

        self.turn_index = turn['turn-index']
        self.transcription = ''

        if labels is not None:
            self.transcription = labels.get('transcription', None)
            self.input = Input(
                turn['input'],
                labels['goal-labels'],
                labels['requested-slots'],
                labels['method-label']
            )
        else:
            self.input = Input(turn['input'], None, None)

        self.output = Output(turn['output'])

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Turn #%d:\n' % self.turn_index
        repr_str += ((indent + step) * ' ' + 'transcription: {0!s}\n'.format(
                     self.transcription))
        repr_str += self.input.pretty_print(indent + step, step)
        repr_str += self.output.pretty_print(indent + step, step)
        repr_str += (indent + step) * ' ' + '\n'
        return repr_str

    def __str__(self):
        return self.pretty_print()


class Input(object):
    """Input from the user.

    Representation of the information dialog manager has about what the
    user said. Contains asr and slu hypotheses.

    Attributes:
        live_asr: A list of asr hypothesis from live system.
        live_slu: A list of slu hypothesis from live system.
        batch_asr: A list of asr hypothesis from batch processing.
        batch_slu: A list of slu hypothesis from batch processing.
    """

    def __init__(self, input_json, user_goal, requested_slots, method):
        """Initialises an input object from the external format.

        Keyword arguments:
            input_json: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """
        self.live_asr = []
        self.live_asr_wcn = []
        self.live_slu = []
        self.batch_asr = []
        self.batch_asr_wcn = []
        self.batch_slu = []

        self.user_goal = user_goal
        self.requested_slots = requested_slots
        self.method = method

        for fldname, asr_field, asr_wcn_field, slu_field in (
                ('live', self.live_asr, self.live_asr_wcn, self.live_slu),
                ('batch', self.batch_asr, self.batch_asr_wcn, self.batch_slu)
            ):
            if fldname in input_json:
                for asr_hyp in input_json[fldname]['asr-hyps']:
                    asr_field.append(ASRHyp(hyp=asr_hyp['asr-hyp'],
                                            score=asr_hyp['score']))

                for wcn_item in input_json[fldname].get('cnet', []):
                    words, scores = zip(*[(item['word'], item['score']) for item in wcn_item['arcs']])
                    asr_wcn_field.append(ASRWCNHyp(hyps=words, scores=scores))

                #import ipdb; ipdb.set_trace()
                #slu_hyps = input_json[fldname]['slu-hyps']

                #slu_scores = [hyp['score'] for hyp in slu_hyps]

                """for hyp_idx, slu_hyp in enumerate(slu_hyps):
                    dialog_acts = []
                    score = slu_scores[hyp_idx]

                    for dialog_act in slu_hyp['slu-hyp']:
                        act = dialog_act['act']
                        slots = []
                        slots_dict = {}
                        da_slots = set()

                        for slot in dialog_act['slots']:
                            slot_name = slot[0]
                            slot_value = str(slot[1]).lower()

                            da_slots.add(slot_name)

                            slots.append(Slot(name=slot_name,
                                              value=slot_value))
                            slots_dict[slot_name] = slot_value

                        dialog_acts.append(DialogAct(act=act,
                                                     slots=tuple(slots)))

                    slu_field.append(SLUHyp(score=score,
                                            acts=dialog_acts))"""

    @property
    def all_slu(self):
        return self.live_slu + self.batch_slu

    @property
    def all_asr(self):
        return self.live_asr + self.batch_asr

    @property
    def all_slots(self):
        slots = []
        for slu_hyp in self.all_slu:
            for da in slu_hyp.acts:
                if da.act == 'inform':
                    slots.extend(da.slots)
        return slots

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Input:\n'
        repr_str += (indent + step) * ' ' + 'Live ASR:\n'
        for asr_hyp in self.live_asr:
            repr_str += (indent + 2 * step) * ' ' + repr(asr_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Live ASR WCN:\n'
        for asr_hyp in self.live_asr_wcn:
            repr_str += (indent + 2 * step) * ' ' + repr(asr_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Live SLU:\n'
        for slu_hyp in self.live_slu:
            repr_str += (indent + 2 * step) * ' ' + repr(slu_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Batch ASR:\n'
        for asr_hyp in self.batch_asr:
            repr_str += (indent + 2 * step) * ' ' + repr(asr_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Batch ASR WCN:\n'
        for asr_hyp in self.batch_asr_wcn:
            repr_str += (indent + 2 * step) * ' ' + repr(asr_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Batch SLU:\n'
        for slu_hyp in self.batch_slu:
            repr_str += (indent + 2 * step) * ' ' + repr(slu_hyp) + '\n'

        return repr_str

    def str(self):
        return self.pretty_print()

    def DiscreteFact__(self):
        return self.pretty_print()


class Output(object):
    """Input for the dialog manager.

    Attributes:
        transcript: Transcript of the output.
        dialog_acts: A list of dialog acts.
    """

    def __init__(self, output_json):
        if 'dialog-acts' in output_json:
            self.transcript = output_json['transcript']
            self.dialog_acts = []
            for act in output_json['dialog-acts']:
                slots = []
                for slot in act['slots']:
                    # coerce the value to a string and lowercase it
                    slot[1] = str(slot[1]).lower()

                    slots.append(Slot(name=slot[0],
                                      value=slot[1]))
                self.dialog_acts.append(DialogAct(act=act['act'],
                                                  slots=slots))
        else:
            self.transcript = ''
            self.dialog_acts = []

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Output:\n'

        repr_str += (indent + step) * ' ' + 'Transcript:\n'
        repr_str += (indent + 2 * step) * ' ' + self.transcript + '\n'

        repr_str += (indent + step) * ' ' + 'Acts:\n'
        for act in self.dialog_acts:
            repr_str += (indent + 2 * step) * ' ' + repr(act) + '\n'

        return repr_str

    def __str__(self):
        return self.pretty_print()


def parse_dialog_from_directory(dialog_dir):
    """
    Keyword arguments:
        dialog_dir: the directory immediately containing the dialogue JSON logs
        regress_to_dais: whether to regress DA scores to scores of single DAIs
        norm_slu_scores: whether scores for SLU hypotheses should be
                         normalised to the scale [0, 1]
        slot_normaliser: instance of a normaliser with normalise method
        reranker_model: if given, an SLU reranker will be applied, using the
                        trained model whose file name is passed in this
                        argument

    """
    log = json.load(open(os.path.join(dialog_dir, 'log.json')))

    labels_file_name = os.path.join(dialog_dir, 'label.json')
    if os.path.exists(labels_file_name):
        labels = json.load(open(labels_file_name))
    else:
        labels = None

    d = Dialog(log, labels)

    return d


ontology = {
    "method": [
        "byconstraints",
        "byname",
        "finished",
        "byalternatives"
    ],
    "food": [
            "afghan",
            "african",
            "afternoon tea",
            "asian oriental",
            "australasian",
            "australian",
            "austrian",
            "barbeque",
            "basque",
            "belgian",
            "bistro",
            "brazilian",
            "british",
            "canapes",
            "cantonese",
            "caribbean",
            "catalan",
            "chinese",
            "christmas",
            "corsica",
            "creative",
            "crossover",
            "cuban",
            "danish",
            "eastern european",
            "english",
            "eritrean",
            "european",
            "french",
            "fusion",
            "gastropub",
            "german",
            "greek",
            "halal",
            "hungarian",
            "indian",
            "indonesian",
            "international",
            "irish",
            "italian",
            "jamaican",
            "japanese",
            "korean",
            "kosher",
            "latin american",
            "lebanese",
            "light bites",
            "malaysian",
            "mediterranean",
            "mexican",
            "middle eastern",
            "modern american",
            "modern eclectic",
            "modern european",
            "modern global",
            "molecular gastronomy",
            "moroccan",
            "new zealand",
            "north african",
            "north american",
            "north indian",
            "northern european",
            "panasian",
            "persian",
            "polish",
            "polynesian",
            "portuguese",
            "romanian",
            "russian",
            "scandinavian",
            "scottish",
            "seafood",
            "singaporean",
            "south african",
            "south indian",
            "spanish",
            "sri lankan",
            "steakhouse",
            "swedish",
            "swiss",
            "thai",
            "the americas",
            "traditional",
            "turkish",
            "tuscan",
            "unusual",
            "vegetarian",
            "venetian",
            "vietnamese",
            "welsh",
            "world"
        ],
        "pricerange": [
            "cheap",
            "moderate",
            "expensive"
        ],
        "name": [
            "ali baba",
            "anatolia",
            "ask",
            "backstreet bistro",
            "bangkok city",
            "bedouin",
            "bloomsbury restaurant",
            "caffe uno",
            "cambridge lodge restaurant",
            "charlie chan",
            "chiquito restaurant bar",
            "city stop restaurant",
            "clowns cafe",
            "cocum",
            "cote",
            "cotto",
            "curry garden",
            "curry king",
            "curry prince",
            "curry queen",
            "da vinci pizzeria",
            "da vince pizzeria",
            "darrys cookhouse and wine shop",
            "de luca cucina and bar",
            "dojo noodle bar",
            "don pasquale pizzeria",
            "efes restaurant",
            "eraina",
            "fitzbillies restaurant",
            "frankie and bennys",
            "galleria",
            "golden house",
            "golden wok",
            "gourmet burger kitchen",
            "graffiti",
            "grafton hotel restaurant",
            "hakka",
            "hk fusion",
            "hotel du vin and bistro",
            "india house",
            "j restaurant",
            "jinling noodle bar",
            "kohinoor",
            "kymmoy",
            "la margherita",
            "la mimosa",
            "la raza",
            "la tasca",
            "lan hong house",
            "little seoul",
            "loch fyne",
            "mahal of cambridge",
            "maharajah tandoori restaurant",
            "meghna",
            "meze bar restaurant",
            "michaelhouse cafe",
            "midsummer house restaurant",
            "nandos",
            "nandos city centre",
            "panahar",
            "peking restaurant",
            "pipasha restaurant",
            "pizza express",
            "pizza express fen ditton",
            "pizza hut",
            "pizza hut city centre",
            "pizza hut cherry hinton",
            "pizza hut fen ditton",
            "prezzo",
            "rajmahal",
            "restaurant alimentum",
            "restaurant one seven",
            "restaurant two two",
            "rice boat",
            "rice house",
            "riverside brasserie",
            "royal spice",
            "royal standard",
            "saffron brasserie",
            "saigon city",
            "saint johns chop house",
            "sala thong",
            "sesame restaurant and bar",
            "shanghai family restaurant",
            "shiraz restaurant",
            "sitar tandoori",
            "stazione restaurant and coffee bar",
            "taj tandoori",
            "tandoori palace",
            "tang chinese",
            "thanh binh",
            "the cambridge chop house",
            "the copper kettle",
            "the cow pizza kitchen and bar",
            "the gandhi",
            "the gardenia",
            "the golden curry",
            "the good luck chinese food takeaway",
            "the hotpot",
            "the lucky star",
            "the missing sock",
            "the nirala",
            "the oak bistro",
            "the river bar steakhouse and grill",
            "the slug and lettuce",
            "the varsity restaurant",
            "travellers rest",
            "ugly duckling",
            "venue",
            "wagamama",
            "yippee noodle bar",
            "yu garden",
            "zizzi cambridge"
        ],
        "area": [
            "centre",
            "north",
            "west",
            "south",
            "east"
        ]
}

def inline_print(string):
    sys.stderr.write('\r\t%s' % (string))
    sys.stderr.flush()


def load_dialogs(data_dir, flists):
    dialog_dirs = get_dialog_dirs(data_dir, flists)

    for i, dialog_dir in enumerate(dialog_dirs):
        inline_print("Loading dialogs: %d/%d" % (i + 1, len(dialog_dirs)))
        dialog = parse_dialog_from_directory(dialog_dir)

        yield dialog


def get_dialog_dirs(data_dir, flists):
    dialog_dirs = []
    for flist in flists:
        with open(flist) as f_in:
            for f_name in f_in:
                dialog_dirs.append(os.path.join(data_dir, f_name.strip()))
    return dialog_dirs
