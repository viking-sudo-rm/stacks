from allennlp.data.dataset_readers.ccgbank import CcgBankDatasetReader
from collections import defaultdict
from tqdm import tqdm


def get_instances(pos_type):
    reader = CcgBankDatasetReader(feature_labels=["ccg", pos_type])
    # train = reader.read("/home/willm/data/ccg/train.txt")
    # test = reader.read("/home/willm/data/ccg/test.txt")
    dev = reader.read("/home/willm/data/ccg/dev.txt")
    return dev
    # return train + test + dev


def main():
    pos_type = "modified_pos"
    instances = get_instances(pos_type)

    # Flatten lambda always seems backwards to me :(
    ccg_tags = [tag for instance in instances for tag in instance["ccg_tags"]]
    pos_tags = [tag for instance in instances for tag in instance[pos_type + "_tags"]]
    
    tag_map = defaultdict(set)
    print("Building tag map")
    for ccg_tag, pos_tag in tqdm(list(zip(ccg_tags, pos_tags))):
        tag_map[pos_tag].add(ccg_tag)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
