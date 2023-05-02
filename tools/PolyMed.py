import json
from tools.data_utilities import Data_variable, Data_integrity

class PolyMed():
    org_case_data = {}
    org_kb_data = {}

    def __init__(self,
                 data_dir: str,
                 plot_dir: str = None,
                 data_type: str = 'extend',
                 display: bool = False,
                 plotting: bool = False,
                 integrity: bool = False
                 ):
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        self.data_type = data_type

        self.__load_data()

        self.data_variable = Data_variable(self.plot_dir, self.data_type)
        self.data_variable.save_data_stat(self.org_case_data, save_type='case', display=display, plotting=plotting)
        self.data_variable.save_data_stat(self.org_kb_data, save_type='knowledge', display=display, plotting=plotting)


        if integrity: self.data_integrity_test()


    def __load_data(self):
        # Diagnosis case data load
        for d_name in ['train', 'test_single', 'test_unseen', 'test_multi']:
            with open(f'{self.data_dir}/eng_{d_name}.json', 'r') as f:
                self.org_case_data[d_name] = json.load(f)[d_name]

        # Knowledge data load
        with open(f'{self.data_dir}/eng_external_medical_knowledge.json', 'r') as f:
            self.org_kb_data = json.load(f)


    def data_integrity_test(self):
        result_dict = {}
        greeting = '==== Knowledge data integrity test ===='
        print(greeting)
        for f_k in self.org_case_data.keys():
            print(f'*{f_k} data integrity test: ')
            print("Completeness test passed: ", Data_integrity.completeness_test(self.org_case_data[f_k]))
            print("Consistency test passed: ", Data_integrity.consistency_test(self.org_case_data[f_k]))
            print("Duplication test passed: ", Data_integrity.duplication_test(self.org_case_data[f_k]))
            print("Data format test passed: ", Data_integrity.data_format_test(self.org_case_data[f_k]))
            print('-'*len(greeting))

        print(f'*Knowledge data integrity test: ')
        print("Completeness test passed: ", Data_integrity.kb_completeness_test(self.org_kb_data))
        print("Consistency test passed: ", Data_integrity.kb_consistency_test(self.org_kb_data))
        print("Duplication test passed: ", Data_integrity.kb_duplication_test(self.org_kb_data))
        print("Data format test passed: ", Data_integrity.kb_data_format_test(self.org_kb_data))
        print('-'*len(greeting))
        print('=' * len(greeting))
