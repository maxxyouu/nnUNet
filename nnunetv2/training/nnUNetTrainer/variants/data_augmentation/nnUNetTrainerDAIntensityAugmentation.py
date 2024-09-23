from typing import Union, Tuple, List

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import numpy as np
from scipy.ndimage import convolve
from PIL import Image

# for custom data augmentation
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
import torch
import torch.nn.functional as F
import SimpleITK as sitk

class IntensityAugmentationTransform(ImageOnlyTransform):
    def __init__(self, linear_factor=0.5, window_size=20):
        super().__init__()

        self.linear_factor = linear_factor
        self.window_size = window_size

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def intensity_augmentation(self, image_array: torch.Tensor, **params) -> torch.Tensor:
        """
        Apply intensity remapping augmentation to the input image tensor
        and return the result as a PIL Image.
        """
        # Generate random noise curve in PyTorch
        random_noise = torch.rand(256) * 255

        # Smooth the random noise curve with a moving average filter (convolution)
        kernel = torch.ones(self.window_size) / self.window_size
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions for 1D convolution
        random_noise = random_noise.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        smoothed_noise = F.conv1d(random_noise, kernel, padding='reflect').squeeze() # use 'reflect' to ensure the same dimension as the input

        # Add a random linear component
        linear_curve_random_factor = torch.choice(torch.tensor([-1, 1]))
        remapping_curve = smoothed_noise + self.linear_factor * linear_curve_random_factor * torch.arange(256)

        # Scale the remapping curve between 0 and 255
        remapping_curve = torch.interp(remapping_curve, torch.tensor([remapping_curve.min(), remapping_curve.max()]), torch.tensor([0, 255]))

        # Create x-coordinates for the remapping curve
        xp = torch.linspace(0, 255, 256)

        # Apply intensity remapping to the 3D medical image (per channel)
        augmented_image = torch.empty_like(image_array)
        for c in range(image_array.shape[0]):  # Iterate over channels
            augmented_image[c] = torch.interp(image_array[c].float(), xp, remapping_curve)

        # # Convert augmented_image tensor to a NumPy array (since SimpleITK works with NumPy arrays)
        # augmented_image_np = augmented_image.cpu().numpy()

        # # Convert the NumPy array to a SimpleITK image
        # # Ensure the order of dimensions is correct for saving, typically [D, H, W, C]
        # sitk_image = sitk.GetImageFromArray(augmented_image_np.transpose(1, 2, 3, 0))  # Transpose from [C, D, H, W] to [D, H, W, C]

        # # Save the image in the .mha format
        # sitk.WriteImage(sitk_image, output_path)

        return augmented_image

class nnUNetTrainer_DefaultDAAndIA(nnUNetTrainer):

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # do the intensity augmentation before anything else
        #NOTE: intensity augmentation is added here.
        transforms.append(RandomTransform(
            IntensityAugmentationTransform(),
            apply_probability=0.5 # NOTE: to be tuned
        ))

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainer_IAOnly(nnUNetTrainer):
    """apply only the intensity augmentation"""

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # do the intensity augmentation before anything else
        #NOTE: intensity augmentation is added here.
        transforms.append(RandomTransform(
            IntensityAugmentationTransform(),
            apply_probability=0.5 # NOTE: to be tuned
        ))

        # transforms.append(RandomTransform(
        #     GaussianNoiseTransform(
        #         noise_variance=(0, 0.1),
        #         p_per_channel=1,
        #         synchronize_channels=True
        #     ), apply_probability=0.1
        # ))
        # transforms.append(RandomTransform(
        #     GaussianBlurTransform(
        #         blur_sigma=(0.5, 1.),
        #         synchronize_channels=False,
        #         synchronize_axes=False,
        #         p_per_channel=0.5, benchmark=True
        #     ), apply_probability=0.2
        # ))
        # transforms.append(RandomTransform(
        #     MultiplicativeBrightnessTransform(
        #         multiplier_range=BGContrast((0.75, 1.25)),
        #         synchronize_channels=False,
        #         p_per_channel=1
        #     ), apply_probability=0.15
        # ))
        # transforms.append(RandomTransform(
        #     ContrastTransform(
        #         contrast_range=BGContrast((0.75, 1.25)),
        #         preserve_range=True,
        #         synchronize_channels=False,
        #         p_per_channel=1
        #     ), apply_probability=0.15
        # ))
        # transforms.append(RandomTransform(
        #     SimulateLowResolutionTransform(
        #         scale=(0.5, 1),
        #         synchronize_channels=False,
        #         synchronize_axes=True,
        #         ignore_axes=ignore_axes,
        #         allowed_channels=None,
        #         p_per_channel=0.5
        #     ), apply_probability=0.25
        # ))
        # transforms.append(RandomTransform(
        #     GammaTransform(
        #         gamma=BGContrast((0.7, 1.5)),
        #         p_invert_image=1,
        #         synchronize_channels=False,
        #         p_per_channel=1,
        #         p_retain_stats=1
        #     ), apply_probability=0.1
        # ))
        # transforms.append(RandomTransform(
        #     GammaTransform(
        #         gamma=BGContrast((0.7, 1.5)),
        #         p_invert_image=0,
        #         synchronize_channels=False,
        #         p_per_channel=1,
        #         p_retain_stats=1
        #     ), apply_probability=0.3
        # ))

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        # default to false
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        # default to false
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


    # def intensity_augmentation(self, image_array):
    #     """
    #     Apply intensity remapping augmentation to the input image array
    #     and save the result as a new PNG image.
    #     NOTE: ORIGINAL NUMPY IMPLEMENTATION
    #     """
    #     # Generate random noise curve
    #     random_noise = np.random.uniform(0, 255, size=256)

    #     # Smooth the random noise curve with a moving average filter
    #     kernel = np.ones(self.window_size) / self.window_size
    #     smoothed_noise = convolve(random_noise, kernel, mode='reflect')

    #     # Add a random linear component
    #     linear_curve_random_factor = np.random.choice([-1, 1])
    #     remapping_curve = smoothed_noise + self.linear_factor * linear_curve_random_factor * np.arange(256)

    #     # Scale the remapping curve between 0 and 255
    #     remapping_curve = np.interp(remapping_curve, (remapping_curve.min(), remapping_curve.max()), (0, 255))

    #     # Create x-coordinates for the remapping curve
    #     xp = np.linspace(0, 255, 256)

    #     # Apply intensity remapping to the image
    #     augmented_image = np.interp(image_array, xp, remapping_curve.astype(int))

    #     rgb_array = np.stack((augmented_image,) * 3, axis=-1)

    #     # Convert the RGB array back to an image
    #     augmented_image = Image.fromarray(rgb_array.astype(np.uint8))

    #     return augmented_image