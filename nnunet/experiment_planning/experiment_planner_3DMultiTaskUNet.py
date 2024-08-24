from copy import deepcopy

import numpy as np
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.paths import *
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet


class ExperimentPlanner3DMultiTaskUNet(ExperimentPlanner3D_v21):

    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3DMultiTaskUNet, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1"# "nnUNetData_FabiansResUNet_v2.1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlans_multiTaskUNet_plans_3D.pkl")
    
    