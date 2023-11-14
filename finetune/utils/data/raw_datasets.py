# Part of the code is adapted from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py
from datasets import load_dataset


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        print("init PromptRawDataset with dataname {}".format(dataset_name))
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        print("init DahoasRMStaicDataset")
        super().__init__(output_path, seed, local_rank, dataset_name)
        print("loading at dataset_name {}".format(dataset_name))
        self.raw_datasets = load_dataset(
            "parquet",
            data_files={
                "train": dataset_name
                + "/data/train-00000-of-00001-2a1df75c6bce91ab.parquet",
                "test": dataset_name
                + "/data/test-00000-of-00001-8c7c51afc6d45980.parquet",
            },
        )
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"
        print("init rm-static dataset finished")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


class LocalJsonFileDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        self.raw_datasets = load_dataset(
            "json",
            data_files={
                "train": chat_path + "/data/train.json",
                "eval": chat_path + "/data/eval.json",
            },
        )

    def get_train_data(self):
        if self.raw_datasets["train"] is not None:
            return self.raw_datasets["train"]
        return None

    def get_eval_data(self):
        if self.raw_datasets["eval"] is not None:
            return self.raw_datasets["eval"]
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample["prompt"] is not None:
            return " " + sample["prompt"]
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        if sample["chosen"] is not None:
            return " " + sample["chosen"]
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        if sample["rejected"] is not None:
            return " " + sample["rejected"]
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["prompt"] is not None and sample["chosen"] is not None:
            return " " + sample["prompt"] + " " + sample["chosen"]
        return None

    def get_prompt_and_rejected(self, sample):
        if sample["prompt"] is not None and sample["rejected"] is not None:
            return " " + sample["prompt"] + " " + sample["rejected"]
        return None


class YiDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        print("data path is {}".format(chat_path))
        self.dataset_name = "yi"
        self.dataset_name_clean = "yi"
        self.raw_datasets = load_dataset(
            "json",
            data_files={
                "train": chat_path + "/data/train.jsonl",
                "eval": chat_path + "/data/eval.jsonl",
            },
        )

    def get_train_data(self):
        if self.raw_datasets["train"] is not None:
            return self.raw_datasets["train"]
        return None

    def get_eval_data(self):
        if self.raw_datasets["eval"] is not None:
            return self.raw_datasets["eval"]
        return None

    def get_prompt(self, sample):
        if sample["prompt"] is not None:
            return " " + sample["prompt"]
        return None

    def get_prompt_and_chosen(self, sample):
        if sample["prompt"] is not None and sample["chosen"] is not None:
            return " " + sample["prompt"] + " " + sample["chosen"]
        return None
