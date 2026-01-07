from enum import Enum

class Config(Enum):
    PROJECT_DIR = '/projects/prjs1531/'
    TRAIN_PATH = '/projects/prjs1531/baidu-ultr-pretrain/'
    TEST_PATH = '/projects/prjs1531/baidu-ultr-pretrain/test/'
    EVALUATION_PATH = '/projects/prjs1531/evaluations'
    CHECKPOINT_PATH = '/projects/prjs1531/checkpoints'

    MULTI_TASK_MODEL = 'multi-task-model-final.pt'
    CLICK_MODEL = 'click-model-final.pt'

    MULTI_TASK_EVALUATION = 'multi-task_evaluation_scores.csv'
    CLICK_EVALUATION = 'click_evaluation_scores.csv'

    MULTI_TASK_FUSE_EVALUATION = 'fused_evaluation_scores.csv'
