import math
import torch
from torch.utils.data._utils.collate import default_collate


DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_id': ' ', 
    'icode': ' ',
}

DEFAULT_NO_PADDING = {
    'origin',
}

class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def apply_patch_to_tensor(x_full, x_patch, patch_idx):
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full
class PaddingCollate_unmerged(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]
    def __call__(self, data_list):
        # Calculate max length for each tag individually
        max_lengths = {
            'heavy': max([data['heavy'][self.length_ref_key].size(0) for data in data_list]),
            'light': max([data['light'][self.length_ref_key].size(0) for data in data_list]),
            'antigen': max([data['antigen'][self.length_ref_key].size(0) for data in data_list])
        }

        # Adjust max lengths to be multiples of 8 if required
        if self.eight:
            max_lengths = {k: math.ceil(v / 8) * 8 for k, v in max_lengths.items()}

        data_list_padded = []

        # Process each data entry in data_list
        for data in data_list:
            data_padded = {}

            # Handle each tag (heavy, light, antigen) separately
            for tag in ['heavy', 'light', 'antigen']:
                # Get common keys for the current tag across all data entries
                keys = self._get_common_keys([d[tag] for d in data_list])

                # Pad each key's content to the max length for the current tag
                data_padded[tag] = {
                    k: self._pad_last(data[tag][k], max_lengths[tag], value=self._get_pad_value(k))
                    if k not in self.no_padding else data[tag][k]
                    for k in data[tag] if k in keys
                }

                # Add mask
                data_padded[tag]['mask'] = self._get_pad_mask(data[tag][self.length_ref_key].size(0), max_lengths[tag])

            data_list_padded.append(data_padded)

        return default_collate(data_list_padded)

    # def __call__(self, data_list):
    #     # Calculate max length across heavy, light, and antigen
    #     max_lengths = {
    #         'heavy': max([data['heavy'][self.length_ref_key].size(0) for data in data_list]),
    #         'light': max([data['light'][self.length_ref_key].size(0) for data in data_list]),
    #         'antigen': max([data['antigen'][self.length_ref_key].size(0) for data in data_list])
    #     }

    #     # Get the overall maximum length across all three tags
    #     overall_max_length = max(max_lengths.values())

    #     # If we need to align to a multiple of 8, adjust the overall maximum length
    #     if self.eight:
    #         overall_max_length = math.ceil(overall_max_length / 8) * 8

    #     data_list_padded = []

    #     # Process each data entry in data_list
    #     for data in data_list:
    #         data_padded = {}

    #         # Handle each tag (heavy, light, antigen)
    #         for tag in ['heavy', 'light', 'antigen']:
    #             # Get common keys for the current tag across all data entries
    #             keys = self._get_common_keys([d[tag] for d in data_list])

    #             # Pad each key's content to the overall max length
    #             data_padded[tag] = {
    #                 k: self._pad_last(data[tag][k], overall_max_length, value=self._get_pad_value(k))
    #                 if k not in self.no_padding else data[tag][k]
    #                 for k in data[tag] if k in keys
    #             }

    #             # Add mask
    #             data_padded[tag]['mask'] = self._get_pad_mask(data[tag][self.length_ref_key].size(0), overall_max_length)

    #         data_list_padded.append(data_padded)

    #     return default_collate(data_list_padded)


class PaddingCollate_unmerged_with_id(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        # Calculate max length for each tag individually
        max_lengths = {
            'heavy': max([data['heavy'][self.length_ref_key].size(0) for data in data_list]),
            'light': max([data['light'][self.length_ref_key].size(0) for data in data_list]),
            'antigen': max([data['antigen'][self.length_ref_key].size(0) for data in data_list])
        }

        # Adjust max lengths to be multiples of 8 if required
        if self.eight:
            max_lengths = {k: math.ceil(v / 8) * 8 for k, v in max_lengths.items()}

        data_list_padded = []

        # Process each data entry in data_list
        for data in data_list:
            data_padded = {}

            # Handle each tag (heavy, light, antigen) separately
            for tag in ['heavy', 'light', 'antigen']:
                # Get common keys for the current tag across all data entries
                keys = self._get_common_keys([d[tag] for d in data_list])

                # Pad each key's content to the max length for the current tag
                data_padded[tag] = {
                    k: self._pad_last(data[tag][k], max_lengths[tag], value=self._get_pad_value(k))
                    if k not in self.no_padding else data[tag][k]
                    for k in data[tag] if k in keys
                }

                # Add mask
                data_padded[tag]['mask'] = self._get_pad_mask(data[tag][self.length_ref_key].size(0), max_lengths[tag])

            # Add the ID as a key in the padded data
            data_padded['id'] = data['id']  # Assuming each data entry has an 'id' field

            data_list_padded.append(data_padded)

        return default_collate(data_list_padded)


