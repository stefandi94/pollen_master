import os.path as osp

RANDOM_STATE = 42
POLLEN_TYPES = ['Dactylus', 'Cynodon', 'Corylus', 'Alnus', 'Betula', 'Salix', 'Fraxinus',
                'Populus', 'Acer', 'Artemisia', 'Taxus', 'Quercus', 'Picea', 'Cedrus']
NUM_OF_CLASSES = len(POLLEN_TYPES)

# DIRECTORIES
# =======================================================================================================================

# BASE DIRECTORY
BASE_DIR = osp.dirname(osp.realpath(__file__))

DATA_DIR = osp.join(BASE_DIR, 'data')

RAW_DATA_DIR = osp.join(DATA_DIR, 'new_raw_data')
NS_RAW_DATA_DIR = RAW_DATA_DIR

NEW_RAW_DATA = osp.join(DATA_DIR, 'new_raw_data')
NEW_DATA = osp.join(DATA_DIR, 'new_data_test_split')

# NS_DATA_DIR = osp.join(EXTRACTED_DATA_DIR, 'NS')
NS_DATA_DIR = NEW_DATA

NS_TRAIN_DIR = osp.join(NS_DATA_DIR, 'train')
NS_VALID_DIR = osp.join(NS_DATA_DIR, 'valid')
NS_TEST_DIR = osp.join(NS_DATA_DIR, 'test')

NS_NORMALIZED_TRAIN_DIR = osp.join(NS_TRAIN_DIR, 'normalized_data')
NS_STANDARDIZED_TRAIN_DIR = osp.join(NS_TRAIN_DIR, 'standardized_data')

NS_NORMALIZED_VALID_DIR = osp.join(NS_VALID_DIR, 'normalized_data')
NS_STANDARDIZED_VALID_DIR = osp.join(NS_VALID_DIR, 'standardized_data')

NS_NORMALIZED_TEST_DIR = osp.join(NS_TEST_DIR, 'normalized_data')
NS_STANDARDIZED_TEST_DIR = osp.join(NS_TEST_DIR, 'standardized_data')

WEIGHTS_DIR = osp.join(BASE_DIR, 'weights')
NS_WEIGHTS_DIR = osp.join(WEIGHTS_DIR, 'ns')

NS_NORM_WEIGHTS_DIR = osp.join(NS_WEIGHTS_DIR, 'normalized')
NS_STAND_WEIGHTS_DIR = osp.join(NS_WEIGHTS_DIR, 'standard_normal')

MODEL_DIR = osp.join(BASE_DIR, 'model_weights')

########################################################################################################################
NS_LIFE_STAT_COMP = osp.join(NS_TRAIN_DIR, 'life_1_stat_comp')
NS_SCATTER_STAT_COMP = osp.join(NS_TRAIN_DIR, 'spectrum_stat_comp')
NS_SIZE_STAT_COMP = osp.join(NS_TRAIN_DIR, 'scatter_stat_comp')

LABEL_DICT_NS = osp.join(BASE_DIR, 'utils', 'label_dict_ns.json')
