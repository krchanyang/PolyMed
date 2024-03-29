from tools.PolyMed import PolyMed
from tools.data_utilities import Training_data
from typing import Optional


class PolymedDataset:
    def __init__(
        self,
        polymed: PolyMed,
        train_data_type: Optional[str],
        test_data_type: Optional[str],
        model_name: str,
        is_tuning: bool = False,
        is_training: bool = True,
    ):
        self.polymed = polymed
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.model_name = model_name
        self.is_tuning = is_tuning
        self.is_training = is_training

    def load_train_data(self):
        if "graph" in self.model_name.lower():
            train_data = Training_data(self.polymed, "train", "graph")
        else:
            train_data = Training_data(self.polymed, "train", "basic")
        if self.train_data_type == "extend" or self.train_data_type == "norm":
            train_x = train_data.train_x
            train_y = train_data.train_y

        if self.train_data_type == "kb_extend":
            print("KB_Extend Train Data Loaded.")

            train_x = train_data.kb_train_x
            train_y = train_data.kb_train_y

        if (
            self.is_tuning is False
            and self.is_training is True
            and "graph" in self.model_name.lower()
        ):
            graph = train_data.graph
            return train_x, train_y, graph

        return train_x, train_y

    def load_test_data(self):
        if "graph" in self.model_name.lower():
            test_data = Training_data(self.polymed, "test", "graph")
        else:
            test_data = Training_data(self.polymed, "test", "basic")

        if self.test_data_type == "single" or self.test_data_type is None:
            if self.train_data_type == "extend" or "norm":
                test_x = test_data.single_test_x
                test_y = test_data.single_test_y
            if self.train_data_type == "kb_extend":
                print("Single KB_Extend Test Data Loaded.")

                test_x = test_data.kb_single_test_x
                test_y = test_data.kb_single_test_y

        if self.test_data_type == "multi":
            if self.train_data_type == "extend" or "norm":
                test_x = test_data.multi_test_x
                test_y = test_data.multi_test_y
            if self.train_data_type == "kb_extend":
                test_x = test_data.kb_multi_test_x
                test_y = test_data.kb_multi_test_y

        if self.test_data_type == "unseen":
            if self.train_data_type == "extend" or "norm":
                test_x = test_data.unseen_test_x
                test_y = test_data.unseen_test_y
            if self.train_data_type == "kb_extend":
                test_x = test_data.kb_unseen_test_x
                test_y = test_data.kb_unseen_test_y

        if (
            self.is_tuning is False
            and self.is_training is False
            and "graph" in self.model_name.lower()
        ):
            graph = test_data.graph
            return test_x, test_y, graph
        return test_x, test_y
