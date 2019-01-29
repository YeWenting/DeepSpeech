import os
import pandas as pd

def preprocess_units(filename="units.txt"):
    all_units_dict = {}
    for line in open(filename):
        wav_filename = line.split()[0] + '.wav'
        new_au_list, au_list = line.split()[1], line.split()[1:]

        # Merge the same symbol
        for idx in range(1, len(au_list)):
            if au_list[idx] != au_list[idx - 1]:
                new_au_list.append(au_list[idx])

        all_units_dict[wav_filename] = new_au_list
    return all_units_dict


def preprocess_zerospeech(root_path, dev_split=0.1):
    # Read and preprocess the unit
    unit_path = os.path.join(root_path, "units.txt")
    print(f"start process {unit_path}...")
    units_dict = preprocess_units(unit_path)

    # Generate the train/dev dataset
    data = []
    datapath = os.path.join(root_path, "train", "unit")
    print(f"Generating train/dev dataset...")
    for wav_filename in os.listdir(datapath):
        wav_filepath = os.path.join(datapath, wav_filename)
        data.append((wav_filename, os.path.getsize(wav_filepath), units_dict[wav_filename]))

    # Train-dev split
    dev_data, train_data = data[:len(data) * dev_split], data[len(data) * dev_split:]
    print(f"train length: {len(train_data)}, dev length: {len(dev_data)}")

    # Save to csv
    for data, name in zip([train_data, dev_data], ['train.csv', 'dev.csv']):
        df = pd.DataFrame.from_records(data=data, columns=["wav_filename", "wav_filesize", "transcript"])
        df.to_csv(os.path.join(root_path, name), index=False)
