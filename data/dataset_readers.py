import os
import json
import numpy as np
from promptsource.templates import DatasetTemplates

from sklearn.metrics import f1_score


def get_dataset_reader(config):
    dataset_class = {
        "scifact": BaseDatasetReader,
        "vc": BaseDatasetReader,
        "fever": BaseDatasetReader,
    }[config.dataset]
    return dataset_class(config)

TRUE_CLASS = ["entailment","SUPPORTS","SUPPORT","SUPPORTED","true","verifiable"]
FALSE_CLASS = ["REFUTED","REFUTES","CONTRADICT","not_entailment","false","unverifiable"]
NEI_CLASS = ["NEI"]

class BaseDatasetReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, config, dataset_stash=("xnli", "en")):
        """
        :param config:
        """
        self.config = config
        self.dataset_stash = dataset_stash
        self.selection = None

        self.templates = DatasetTemplates(*self.dataset_stash)
        self.train_template = self.get_template(self.config.train_template_idx)
        self.eval_template = self.get_template(self.config.eval_template_idx)
        self.save_data_file = config.save_data_file


    def get_template(self, template_idx):
        template_names = self.templates.all_template_names
        if template_idx >= 0:
            return self.templates[template_names[template_idx]]
        elif template_idx == -1:

            list_idx = []
            list_templates = []
            for idx, template_name in enumerate(template_names):
                if self.templates[template_name].metadata.original_task:
                    list_idx.append(idx)
                    list_templates.append(self.templates[template_name])
            print(list_idx)

            return list_templates
        elif template_idx == -2:
            return [self.templates[template_name] for template_name in template_names]

    def get_train_template(self):
        return self.train_template

    def get_eval_template(self):
        return self.eval_template
    
    def load_jsonl(self,filename):
        d_list = []
        with open(filename, encoding='utf-8', mode='r') as in_f:
            print("Load Jsonl:", filename)
            for line in in_f:
                item = json.loads(line.strip())
                d_list.append(item)

        return d_list

    def save_jsonl(self,d_list, filename):
        print("Save to Jsonl:", filename)
        with open(filename, encoding='utf-8', mode='w') as out_f:
            for item in d_list:
                out_f.write(json.dumps(item) + '\n')

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if split == "validation":

            load_file = f"{self.config.dataset_offline}/{self.config.dataset}_{split}.jsonl"
            orig_data = self.load_jsonl(load_file)

            return orig_data[:self.config.sample_num]
           
        load_file = f"{self.config.dataset_offline}/{self.config.dataset}_{split}.jsonl"
        orig_data = self.load_jsonl(load_file)

        return orig_data

    def read_few_shot_dataset(self):

        file_dir = os.path.join(self.config.dataset_offline, "few_shot", self.config.dataset,f"{self.config.num_shot}_shot")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, f"{self.config.few_shot_random_seed}_seed.jsonl")

        if os.path.exists(file_path):
            file_path1 = file_path
            with open(file_path1, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))
            return data
        else:
            orig_data = self.read_orig_dataset("train")
            selected_data = self._sample_few_shot_data(orig_data)

            if self.save_data_file:
                with open(file_path, "w+") as fout:
                    for example in selected_data:
                        fout.write(json.dumps(example) + "\n")
        
            return selected_data
            
    def _sample_few_shot_data(self, orig_data):
        
        saved_random_state = np.random.get_state()
        np.random.seed(self.config.few_shot_random_seed)
        np.random.shuffle(orig_data)

        g1 = [x for x in orig_data if x["label"] in TRUE_CLASS]
        g2 = [x for x in orig_data if x["label"] in FALSE_CLASS]
        g3 = [x for x in orig_data if x["label"] in NEI_CLASS]

        np.random.shuffle(g1)
        np.random.shuffle(g2)
        np.random.shuffle(g3)
        sele_n = self.config.n_ways
        selected_data1 = g1[: int(self.config.num_shot/sele_n)]
        selected_data2 = g2[: int(self.config.num_shot/sele_n)]
        selected_data3 = g3[: int(self.config.num_shot/sele_n)]
        np.random.set_state(saved_random_state)
        selected_data = selected_data1+selected_data2+selected_data3
        return selected_data

    def compute_metric(self, accumulated):
        labels = []
        pre = []

        for i,p in enumerate(accumulated["prediction"]):
            pre.append(p)
            tmp_label = accumulated["labels"][i]
            if isinstance(tmp_label,list):
                tmp_label = tmp_label[0]
            labels.append(tmp_label)

        f1_sc = f1_score(labels, pre, average='macro')
        print(f"Test F1 : {f1_sc}")
        if self.config.num_steps == 0:
            print(self.config.load_weight)
        return {"Test F1": f1_sc}
