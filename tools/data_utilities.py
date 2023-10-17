import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import defaultdict, Counter
from matplotlib.colors import LinearSegmentedColormap


class Data_variable:
    data_id_parser = []
    info_all_data = {}
    uniq_all_data = {}
    accum_all_data = {}
    count_all_data = {}

    info_by_data = {}
    uniq_by_data = {}
    accum_by_data = {}
    count_by_data = {}

    info_kb_data = {}
    uniq_kb_data = {}
    accum_kb_data = {}
    count_kb_data = {}

    word_idx_total = {}
    idx_word_total = {}
    word_idx_case = {}
    idx_word_case = {}
    word_idx_kb = {}
    idx_word_kb = {}
    word_idx_allkb = {}
    idx_word_allkb = {}

    def __init__(self, plot_dir: str, data_type: str):
        self.plot_dir = plot_dir
        self.data_type = data_type

    def gen_index_dict(self, da_type):
        word_idx_total = defaultdict(dict)
        idx_word_total = defaultdict(dict)
        word_idx_case = defaultdict(dict)
        idx_word_case = defaultdict(dict)
        word_idx_kb = defaultdict(dict)
        idx_word_kb = defaultdict(dict)
        word_idx_allkb = defaultdict(dict)
        idx_word_allkb = defaultdict(dict)

        if self.word_idx_total and self.idx_word_total:
            word_idx_total = defaultdict(dict, self.word_idx_total)
            idx_word_total = defaultdict(dict, self.idx_word_total)

        if da_type == "case":
            for key in self.uniq_all_data.keys():
                for item in self.uniq_all_data[key]:
                    if item not in word_idx_total[key].keys():
                        word_idx_total[key][item] = len(word_idx_total[key])
                        idx_word_total[key][len(idx_word_total[key])] = item
                    if item not in word_idx_case[key].keys():
                        word_idx_case[key][item] = len(word_idx_case[key])
                        idx_word_case[key][len(idx_word_case[key])] = item

            self.word_idx_total = dict(word_idx_total)
            self.idx_word_total = dict(idx_word_total)
            self.word_idx_case = dict(word_idx_case)
            self.idx_word_case = dict(idx_word_case)

        if da_type == "knowledge":
            for key in self.uniq_kb_data.keys():
                for item in self.uniq_kb_data[key]:
                    if item not in word_idx_total[key].keys():
                        word_idx_total[key][item] = len(word_idx_total[key])
                        idx_word_total[key][len(idx_word_total[key])] = item
                    if item not in word_idx_kb[key].keys():
                        word_idx_kb[key][item] = len(word_idx_kb[key])
                        idx_word_kb[key][len(idx_word_kb[key])] = item
                    if item not in word_idx_allkb.keys():
                        word_idx_allkb[item] = len(word_idx_allkb)
                        idx_word_allkb[len(idx_word_allkb)] = item

            self.word_idx_total = dict(word_idx_total)
            self.idx_word_total = dict(idx_word_total)
            self.word_idx_kb = dict(word_idx_kb)
            self.idx_word_kb = dict(idx_word_kb)
            self.word_idx_allkb = dict(word_idx_allkb)
            self.idx_word_allkb = dict(idx_word_allkb)

    def gen_case_data_stat(self, org_data, display, plotting):
        data_id_parser = []
        info_all_data = {}
        uniq_all_data = defaultdict(list)
        accum_all_data = defaultdict(list)
        count_all_data = defaultdict(list)

        info_by_data = defaultdict(dict)
        uniq_by_data = {}
        accum_by_data = {}
        count_by_data = {}

        for d_type in org_data.keys():
            temp_by_type = defaultdict(list)
            temp_accum_by_type = defaultdict(list)
            temp_count_by_type = defaultdict(list)
            for ea_data in org_data[d_type]:
                data_id_parser.append(ea_data["id"])

                for norm_comp in ["symptoms", "diagnosis"]:
                    # unique by data
                    temp_by_type[norm_comp] = self.push_data(
                        temp_by_type[norm_comp], ea_data[norm_comp]
                    )

                    # unique by all
                    uniq_all_data[norm_comp] = self.push_data(
                        uniq_all_data[norm_comp], ea_data[norm_comp]
                    )

                    # accum by data
                    temp_accum_by_type[norm_comp] = self.push_data(
                        temp_accum_by_type[norm_comp],
                        ea_data[norm_comp],
                        p_type="duple",
                    )

                    # accum by all
                    accum_all_data[norm_comp] = self.push_data(
                        accum_all_data[norm_comp], ea_data[norm_comp], p_type="duple"
                    )

                    # count by data
                    temp_count_by_type[norm_comp].append(len(ea_data[norm_comp]))

                    # count by all
                    count_all_data[norm_comp].append(len(ea_data[norm_comp]))

                if self.data_type == "extend":
                    for extend_comp in [
                        "sex",
                        "age",
                        "family_history",
                        "background",
                        "underlying_disease",
                    ]:
                        # unique by data
                        temp_by_type[extend_comp] = self.push_data(
                            temp_by_type[extend_comp], ea_data[extend_comp]
                        )

                        # unique by all
                        uniq_all_data[extend_comp] = self.push_data(
                            uniq_all_data[extend_comp], ea_data[extend_comp]
                        )

                        # accum by data
                        temp_accum_by_type[extend_comp] = self.push_data(
                            temp_accum_by_type[extend_comp],
                            ea_data[extend_comp],
                            p_type="duple",
                        )

                        # accum by all
                        accum_all_data[extend_comp] = self.push_data(
                            accum_all_data[extend_comp],
                            ea_data[extend_comp],
                            p_type="duple",
                        )

                        # count by data
                        temp_count_by_type[extend_comp] = self.push_data(
                            temp_count_by_type[extend_comp],
                            ea_data[extend_comp],
                            p_type="len",
                        )

                        # count by all
                        count_all_data[extend_comp] = self.push_data(
                            count_all_data[extend_comp],
                            ea_data[extend_comp],
                            p_type="len",
                        )

            uniq_by_data[d_type] = dict(temp_by_type)
            accum_by_data[d_type] = dict(temp_accum_by_type)
            count_by_data[d_type] = dict(temp_count_by_type)
            info_by_data[d_type]["total_data"] = len(count_by_data[d_type]["symptoms"])

            for temp_k in uniq_by_data[d_type].keys():
                info_by_data[d_type]["num_" + temp_k] = len(
                    uniq_by_data[d_type][temp_k]
                )
                info_by_data[d_type]["mean_" + temp_k] = np.mean(
                    count_by_data[d_type][temp_k]
                )
                none_zero_list = np.ma.masked_equal(count_by_data[d_type][temp_k], 0)
                info_by_data[d_type]["masked_mean_" + temp_k] = np.mean(none_zero_list)
                info_by_data[d_type]["missing_" + temp_k] = (
                    Counter(accum_by_data[d_type][temp_k])[None]
                    / len(count_by_data[d_type][temp_k])
                    if None in Counter(accum_by_data[d_type][temp_k]).keys()
                    else 0
                )

        self.info_by_data = dict(info_by_data)
        self.uniq_by_data = dict(uniq_by_data)
        self.accum_by_data = dict(accum_by_data)
        self.count_by_data = dict(count_by_data)

        info_all_data["total_data"] = len(count_all_data["symptoms"])
        for temp_k in uniq_all_data.keys():
            info_all_data["num_" + temp_k] = len(uniq_all_data[temp_k])
            info_all_data["mean_" + temp_k] = np.mean(count_all_data[temp_k])
            none_zero_list = np.ma.masked_equal(count_all_data[temp_k], 0)
            info_all_data["masked_mean_" + temp_k] = np.mean(none_zero_list)
            info_all_data["missing_" + temp_k] = (
                Counter(accum_all_data[temp_k])[None] / len(count_all_data[temp_k])
                if None in Counter(accum_all_data[temp_k]).keys()
                else 0
            )

        self.data_id_parser = data_id_parser
        self.info_all_data = dict(info_all_data)
        self.uniq_all_data = dict(uniq_all_data)
        self.accum_all_data = dict(accum_all_data)
        self.count_all_data = dict(count_all_data)

        self.gen_index_dict(da_type="case")

        if display:
            self.print_stat(info_all_data, multi=False)
            self.print_stat(info_by_data, multi=True)
        if plotting:
            self.data_plot(accum_all_data, knowledge=False, data_name="case_data")

    def gen_knowledge_stat(self, org_data, display, plotting):
        info_kb_data = {}
        uniq_kb_data = defaultdict(list)
        accum_kb_data = defaultdict(list)
        count_kb_data = defaultdict(list)

        for d_name in org_data.keys():
            uniq_kb_data["diagnosis"] = self.push_data(
                uniq_kb_data["diagnosis"], d_name
            )
            accum_kb_data["diagnosis"] = self.push_data(
                accum_kb_data["diagnosis"], d_name, p_type="duple"
            )
            count_kb_data["diagnosis"] = self.push_data(
                count_kb_data["diagnosis"], d_name, p_type="len"
            )

            for norm_comp in [
                "symptoms",
                "relations",
                "department",
                "location",
                "category",
            ]:
                uniq_kb_data[norm_comp] = self.push_data(
                    uniq_kb_data[norm_comp], org_data[d_name][norm_comp]
                )
                accum_kb_data[norm_comp] = self.push_data(
                    accum_kb_data[norm_comp],
                    org_data[d_name][norm_comp],
                    p_type="duple",
                )
                count_kb_data[norm_comp] = self.push_data(
                    count_kb_data[norm_comp], org_data[d_name][norm_comp], p_type="len"
                )

        info_kb_data["total_data"] = len(count_kb_data["symptoms"])
        for temp_k in uniq_kb_data.keys():
            info_kb_data["num_" + temp_k] = len(uniq_kb_data[temp_k])
            info_kb_data["mean_" + temp_k] = np.mean(count_kb_data[temp_k])
            none_zero_list = np.ma.masked_equal(count_kb_data[temp_k], 0)
            info_kb_data["masked_mean_" + temp_k] = np.mean(none_zero_list)
            info_kb_data["missing_" + temp_k] = (
                Counter(accum_kb_data[temp_k])[None] / len(count_kb_data[temp_k])
                if None in Counter(accum_kb_data[temp_k]).keys()
                else 0
            )

        self.info_kb_data = dict(info_kb_data)
        self.uniq_kb_data = dict(uniq_kb_data)
        self.accum_kb_data = dict(accum_kb_data)
        self.count_kb_data = dict(count_kb_data)

        self.gen_index_dict(da_type="knowledge")

        if display:
            self.print_stat(info_kb_data, multi=False, data_name="Knowledge data")
        if plotting:
            self.data_plot(accum_kb_data, knowledge=True, data_name="knowledge data")

    def save_data_stat(self, org_data, save_type, display=False, plotting=False):
        """
        Define the data statistic variable and assign the values:
         - Data information(All, data by data): sum, mean, num_data of data and data attribute(symptoms, diagnosis, background etc.)
         - Unique data(All, data by data): unique words by each data attributes(symptoms, diagnosis, background etc.)
        """
        if save_type == "knowledge":
            self.gen_knowledge_stat(org_data, display, plotting)
        else:
            self.gen_case_data_stat(org_data, display, plotting)

    def push_data(self, bowl, data, p_type="unique"):
        if p_type == "unique":
            if data == None:
                return bowl
            if type(data) == str or type(data) == int:
                if not data in bowl:
                    bowl.append(data)
            elif type(data) == list:
                for ea_data in data:
                    if not ea_data in bowl:
                        bowl.append(ea_data)

        elif p_type == "duple":
            if data == None:
                bowl.append(None)
            elif type(data) == str or type(data) == int:
                bowl.append(data)
            elif type(data) == list:
                for ea_data in data:
                    bowl.append(ea_data)

        elif p_type == "len":
            if data == None:
                bowl.append(0)
            if type(data) == str or type(data) == int:
                bowl.append(1)
            elif type(data) == list:
                bowl.append(len(data))

        return bowl

    def print_stat(self, info_data, multi, data_name="Total_data"):
        if not multi:
            Data_statistic.data_summary(info_data, data_name, True)
        if multi:
            for ea_info in info_data.keys():
                Data_statistic.data_summary(info_data[ea_info], ea_info, True)

    def data_plot(self, accum_data, knowledge, data_name):
        if not knowledge:
            for key_name, title in zip(
                ["symptoms", "diagnosis", "age", "sex"],
                ["Symptoms", "Disease", "Age", "Sex"],
            ):
                Data_statistic.data_plotting(
                    Counter(accum_data[key_name]).most_common(),
                    title,
                    self.plot_dir,
                    data_name,
                )
        else:
            for key_name, title in zip(
                [
                    "symptoms",
                    "diagnosis",
                    "relations",
                    "department",
                    "location",
                    "category",
                ],
                [
                    "Symptoms",
                    "Disease",
                    "Relations",
                    "Department",
                    "Location",
                    "Category",
                ],
            ):
                Data_statistic.data_plotting(
                    Counter(accum_data[key_name]).most_common(),
                    title,
                    self.plot_dir,
                    data_name,
                )


class Data_statistic:
    @staticmethod
    def data_summary(data_info, data_name, table_filp=False):
        if not table_filp:
            # Get the column names from the data_info dictionary keys
            column_names = list(data_info.keys())
            # Calculate the maximum length of the column names and the data name
            max_name_length = max([len(name) for name in column_names + [data_name]])
            # Create a format string for the table
            format_string = "| {:<{}} " + " | {:^10} " * len(column_names) + "|"
            # Print the table header
            print("=" * (len(format_string) + 20))
            print(
                format_string.format(
                    "name".center(max_name_length), max_name_length, *column_names
                )
            )
            print("-" * (len(format_string) + 20))
            # Print the data row
            print(
                format_string.format(
                    data_name.center(max_name_length),
                    max_name_length,
                    *[
                        "{:.1f}".format(data_info[name]) if name in data_info else ""
                        for name in column_names
                    ],
                )
            )
            print("=" * (len(format_string) + 20))
        else:
            # Get the row names from the data_info dictionary keys
            row_names = list(data_info.keys())
            # Calculate the maximum length of the row names and the data name
            max_name_length = max([len(name) for name in row_names + [data_name]])
            # Create a format string for the table
            title_sting = "|      {:^{}}     |"
            format_string = "| {:<{}} | {:^{}} |"
            # Print the table header
            print("=" * (max_name_length * 2 + 7))
            print(
                title_sting.format(
                    f"{data_name}".center(max_name_length), max_name_length * 2 - 6
                )
            )
            print("=" * (max_name_length * 2 + 7))
            print(
                format_string.format(
                    "State name".center(max_name_length),
                    max_name_length,
                    "result".center(max_name_length),
                    max_name_length,
                )
            )
            print("-" * (max_name_length * 2 + 7))
            # Print the data rows
            for row_name in row_names:
                print(
                    format_string.format(
                        row_name.center(max_name_length),
                        max_name_length,
                        "{:.1f}".format(data_info[row_name]).center(max_name_length),
                        max_name_length,
                    )
                )
            print("=" * (max_name_length * 2 + 7))

    @staticmethod
    def data_plotting(data, data_name, plot_dir=None, data_lname=None):
        # Get the x and y values from the data
        x_values = [i[0] for i in data if i[0] is not None]
        y_values = [i[1] for i in data if i[0] is not None]

        if data_name == "Age":
            re_x_values = []
            re_y_values = []
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            for i in range(10):
                re_y_values.append(
                    np.sum(
                        y_values[
                            np.where(
                                (x_values >= (i * 10) + 1) & (x_values <= (i * 10) + 10)
                            )[0]
                        ]
                    )
                )
                re_x_values.append(f"{(i*10)+1}-{(i*10)+10}")
            x_values = re_x_values
            y_values = re_y_values

        norm = plt.Normalize(min(y_values), max(y_values))

        # Define a color map
        colors = [(0.7, 0.8, 0.9), (102 / 255, 153 / 2555, 255 / 255)]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

        # Create the histogram
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.histplot(
            x=x_values,
            weights=y_values,
            palette=cmap,
            hue=y_values,
            hue_norm=norm,
            element="step",
            fill=True,
            kde=False,
            linewidth=0.5,
            ax=ax,
            legend=False,
        )

        # Set the x-axis labels and rotate the labels for better visibility
        plt.xticks(rotation=45, ha="right")
        ax.set_xlabel(f"{data_name}", fontsize=16, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=16, fontweight="bold")
        ax.set_title(
            f"{data_name} Frequency Distribution", fontsize=20, fontweight="bold"
        )
        if len(x_values) > 20:
            ax.set(xticklabels=[])
        plt.tight_layout()

        # Remove the spines
        sns.despine()

        # Remove the background color
        ax.set_facecolor("none")

        if plot_dir and data_name:
            plt.savefig(f"{plot_dir}/{data_lname}_{data_name}.png", dpi=300)

        plt.show()


class Data_integrity:
    @staticmethod
    def completeness_test(data):
        try:
            for record in data:
                if (
                    record.get("id") is None
                    or record.get("diagnosis") is None
                    or record.get("symptoms") is None
                ):
                    return False
            return True
        except Exception as e:
            print(f"Test was blocked by runtime error and failed: {e}")
            return False

    @staticmethod
    def consistency_test(data):
        try:
            for record in data:
                if record.get("age") is not None and (
                    record.get("age") < 0 or record.get("age") > 200
                ):
                    return False
                if record.get("sex") not in [None, "M", "W"]:
                    return False
            return True
        except Exception as e:
            print(f"Test was blocked by runtime error and failed: {e}")
            return False

    @staticmethod
    def duplication_test(data):
        try:
            for record in data:
                if record.get("family_history") is not None and len(
                    record["family_history"]
                ) != len(set(record["family_history"])):
                    return False
                if record.get("background") is not None and len(
                    record["background"]
                ) != len(set(record["background"])):
                    return False
                if record.get("underlying_disease") is not None and len(
                    record["underlying_disease"]
                ) != len(set(record["underlying_disease"])):
                    return False
                if len(record["diagnosis"]) != len(set(record["diagnosis"])):
                    return False
                if len(record["symptoms"]) != len(set(record["symptoms"])):
                    return False
            return True

        except Exception as e:
            print(f"Test was blocked by runtime error and failed: {e}")
            return False

    @staticmethod
    def data_format_test(data):
        try:
            for record in data:
                if not isinstance(record.get("id"), int):
                    return False
                if not isinstance(record.get("category"), str):
                    return False
                if record.get("sex") not in [None, "M", "W"]:
                    return False
                if record.get("family_history") is not None and not isinstance(
                    record["family_history"], list
                ):
                    return False
                if record.get("background") is not None and not isinstance(
                    record["background"], list
                ):
                    return False
                if record.get("underlying_disease") is not None and not isinstance(
                    record["underlying_disease"], list
                ):
                    return False
                if not isinstance(record["diagnosis"], list):
                    return False
                if not isinstance(record["symptoms"], list):
                    return False
            return True
        except Exception as e:
            print(f"Test was blocked by runtime error and failed: {e}")
            return False

    @staticmethod
    def kb_completeness_test(data):
        for key, value in data.items():
            if "symptoms" not in value:
                return False

        return True

    @staticmethod
    def kb_consistency_test(data):
        for key, value in data.items():
            if "age" in value:
                if value["age"] < 0 or value["age"] > 200:
                    return False
            if "sex" in value:
                if value["sex"] not in ["None", "W", "M"]:
                    return False

        return True

    @staticmethod
    def kb_duplication_test(data):
        for key, value in data.items():
            for field in [
                "symptoms",
                "relations",
                "department",
                "location",
                "category",
            ]:
                if field in value:
                    if len(value[field]) != len(set(value[field])):
                        return False

        return True

    @staticmethod
    def kb_data_format_test(data):
        for key, value in data.items():
            for field in [
                "symptoms",
                "relations",
                "department",
                "location",
                "category",
            ]:
                if field in value:
                    if not isinstance(value[field], list):
                        return False

        return True


class Data_preprocessing:
    @staticmethod
    def to_numpy_onehot(data, word_idx_book, extend="extend", kb_train=False):
        if kb_train:
            word_idx_book = word_idx_book.word_idx_total
        else:
            word_idx_book = word_idx_book.word_idx_case
        multi_on = False
        if extend == "extend":
            temp_x = np.empty(
                (
                    0,
                    len(word_idx_book["symptoms"])
                    + 2
                    + 1
                    + len(word_idx_book["background"])
                    + len(word_idx_book["underlying_disease"])
                    + len(word_idx_book["family_history"]),
                )
            )
        else:
            temp_x = np.empty((0, len(word_idx_book["symptoms"])))
        for ea_data in data:
            if len(ea_data["diagnosis"]) != 1:
                multi_on = True
        if multi_on:
            temp_y = []
        else:
            temp_y = np.empty((0, 1))
        for ea_data in data:
            empty_x = np.zeros(len(word_idx_book["symptoms"]))
            for symptoms in ea_data["symptoms"]:
                empty_x[word_idx_book["symptoms"][symptoms]] = 1
            if extend == "extend":
                sex_x = np.zeros(2)
                if ea_data["sex"] == "남":
                    sex_x[0] = 1
                elif ea_data["sex"] == "여":
                    sex_x[1] = 1

                age_x = (
                    np.array([0])
                    if ea_data["age"] == None
                    else np.array([int(ea_data["age"])])
                )

                background_x = np.zeros(len(word_idx_book["background"]))
                if ea_data["background"] != None:
                    for ea_background in ea_data["background"]:
                        background_x[word_idx_book["background"][ea_background]] = 1

                underlying_disease_x = np.zeros(
                    len(word_idx_book["underlying_disease"])
                )
                if ea_data["underlying_disease"] != None:
                    for ea_underlying_disease in ea_data["underlying_disease"]:
                        underlying_disease_x[
                            word_idx_book["underlying_disease"][ea_underlying_disease]
                        ] = 1

                family_history_x = np.zeros(len(word_idx_book["family_history"]))
                if ea_data["family_history"] != None:
                    for ea_family_history in ea_data["family_history"]:
                        family_history_x[
                            word_idx_book["family_history"][ea_family_history]
                        ] = 1

                empty_x = np.hstack(
                    [
                        empty_x,
                        sex_x,
                        age_x,
                        background_x,
                        underlying_disease_x,
                        family_history_x,
                    ]
                )

            temp_x = np.vstack([temp_x, empty_x])
            if multi_on:
                tt_y = []
                for tt_temp in ea_data["diagnosis"]:
                    tt_y.append(word_idx_book["diagnosis"][tt_temp])
                temp_y.append(tt_y)
            else:
                temp_y = np.vstack(
                    [
                        temp_y,
                        np.array([word_idx_book["diagnosis"][ea_data["diagnosis"][0]]]),
                    ]
                )

        if multi_on:
            return temp_x, temp_y
        else:
            return temp_x, temp_y.reshape(len(temp_y))

    @staticmethod
    def to_numpy_kb_onehot(kb, word_idx_book, extend=False):
        if extend == "extend":
            temp_x = np.empty(
                (
                    0,
                    len(word_idx_book["symptoms"])
                    + 2
                    + 1
                    + len(word_idx_book["background"])
                    + len(word_idx_book["underlying_disease"])
                    + len(word_idx_book["family_history"]),
                )
            )
        else:
            temp_x = np.empty((0, len(word_idx_book["symptoms"])))
        temp_y = np.empty((0, 1))

        for k in kb.keys():
            empty_x = np.zeros(len(word_idx_book["symptoms"]))
            for sym in kb[k]["symptoms"]:
                empty_x[word_idx_book["symptoms"][sym]] = 1

            if extend == "extend":
                sex_x = np.zeros(2)
                age_x = np.array([0])
                bg_x = np.zeros(len(word_idx_book["background"]))
                ud_x = np.zeros(len(word_idx_book["underlying_disease"]))
                fh_x = np.zeros(len(word_idx_book["family_history"]))

                empty_x = np.hstack([empty_x, sex_x, age_x, bg_x, ud_x, fh_x])

            temp_x = np.vstack([temp_x, empty_x])
            temp_y = np.vstack([temp_y, np.array([word_idx_book["diagnosis"][k]])])

        return temp_x, temp_y.reshape(len(temp_y))

    @staticmethod
    def get_kb_relation(kb, word_idx_book):
        src = []
        dst = []
        for f_k in kb.keys():
            for s_k in kb[f_k].keys():
                for src_word in kb[f_k][s_k]:
                    src.extend([word_idx_book[f_k], word_idx_book[src_word]])
                    dst.extend([word_idx_book[src_word], word_idx_book[f_k]])
                    for dst_word in kb[f_k][s_k]:
                        if src_word == dst_word:
                            continue
                        src.append(word_idx_book[src_word])
                        dst.append(word_idx_book[dst_word])

        return src, dst


class Training_data:
    train_x = []
    train_y = []
    single_test_x = []
    single_test_y = []
    unseen_test_x = []
    unseen_test_y = []
    multi_test_x = []
    multi_test_y = []

    kb_train_x = []
    kb_train_y = []
    kb_single_test_x = []
    kb_single_test_y = []
    kb_unseen_test_x = []
    kb_unseen_test_y = []
    kb_multi_test_x = []
    kb_multi_test_y = []

    graph = None

    def __init__(self, polymed, t_type, m_type = "basic"):
        if t_type == "train":
            self.__load_train_data(polymed)
            self.__load_test_data(polymed)
        else:
            self.__load_test_data(polymed)
        if m_type == "graph":
            self.__load_graph_data(polymed)

    def __load_train_data(self, polymed):
        print("Train data load...", end="")
        self.train_x, self.train_y = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["train"],
            polymed.data_variable,
            polymed.data_type,
            False,
        )
        kx, ky = Data_preprocessing.to_numpy_kb_onehot(
            polymed.org_kb_data,
            polymed.data_variable.word_idx_total,
            extend="extend" if polymed.data_type == "extend" else False,
        )
        ktrain_x, ktrain_y = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["train"],
            polymed.data_variable,
            polymed.data_type,
            True,
        )

        self.kb_train_x, self.kb_train_y = np.vstack([ktrain_x, kx]), np.hstack(
            [ktrain_y, ky]
        )

        print("[Done]")

    def __load_test_data(self, polymed):
        print("Test data load...", end="")
        self.single_test_x, self.single_test_y = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["test_single"],
            polymed.data_variable,
            polymed.data_type,
            False,
        )
        self.unseen_test_x, self.unseen_test_y = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["test_unseen"],
            polymed.data_variable,
            polymed.data_type,
            False,
        )
        self.multi_test_x, self.multi_test_y = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["test_multi"],
            polymed.data_variable,
            polymed.data_type,
            False,
        )

        (
            self.kb_single_test_x,
            self.kb_single_test_y,
        ) = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["test_single"],
            polymed.data_variable,
            polymed.data_type,
            True,
        )
        (
            self.kb_unseen_test_x,
            self.kb_unseen_test_y,
        ) = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["test_unseen"],
            polymed.data_variable,
            polymed.data_type,
            True,
        )
        self.kb_multi_test_x, self.kb_multi_test_y = Data_preprocessing.to_numpy_onehot(
            polymed.org_case_data["test_multi"],
            polymed.data_variable,
            polymed.data_type,
            True,
        )

        print("[Done]")

    def __load_graph_data(self, polymed):
        import dgl
        
        print("Graph data load...", end="")
        src, dst = Data_preprocessing.get_kb_relation(
            polymed.org_kb_data, polymed.data_variable.word_idx_allkb
        )
        graph = dgl.graph((src, dst))
        self.graph = dgl.add_self_loop(graph)
        print("[Done]")
