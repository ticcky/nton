import json
import os

from data_dstc2 import parse_dialog_from_directory


def import_dstc_db(data_dir, flists):

    dialog_dirs = []
    for flist in flists:
        with open(flist) as f_in:
            for f_name in f_in:
                dialog_dirs.append(os.path.join(data_dir, f_name.strip()))

    db = {}

    for i, dialog_dir in enumerate(dialog_dirs):
        dialog = parse_dialog_from_directory(dialog_dir)

        for turn in dialog.turns:
            is_offer = False
            obj = {}
            for da in turn.output.dialog_acts:
                if da.act == "offer":
                    is_offer = True
                if is_offer:
                    for slot in da.slots:
                        obj[slot.name] = slot.value

            if is_offer:
                name = obj['name']
                if name in db:
                    for k, v in obj.items():
                        if k in db[name]:
                            if db[name][k] != v:
                                import ipdb; ipdb.set_trace()
                    db[name].update(obj)
                else:
                    db[name] = obj

    entries = db.values()

    for entry in entries:
        for key in ["addr", "area", "food", "phone", "pricerange", "postcode", "name"]:
            if not key in entry:
                entry[key] = "not available"

    return entries


def main(data_dir, out_db_file):
    input_dir = os.path.join(data_dir, 'dstc2/data')
    flist1 = os.path.join(data_dir, 'dstc2/scripts/config/dstc2_train.flist')
    flist2 = os.path.join(data_dir, 'dstc2/scripts/config/dstc2_dev.flist')
    flist3 = os.path.join(data_dir, 'dstc2/scripts/config/dstc2_test.flist')

    db = import_dstc_db(input_dir, [flist1, flist2, flist3])
    with open(out_db_file, 'w') as f_out:
        json.dump(db, f_out, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_db_file', required=True)

    args = parser.parse_args()

    main(**vars(args))