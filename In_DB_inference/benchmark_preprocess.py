

import time
import torch
import time
from typing import Any, List, Dict, Tuple


def pre_processing(mini_batch_data: List[Tuple]):
    """
    mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
    """
    sample_lines = len(mini_batch_data)
    feat_id = []
    feat_value = []
    y = []

    for i in range(sample_lines):
        row_value = mini_batch_data[i]
        sample = decode_libsvm(row_value)
        feat_id.append(sample['id'])
    feat_id = torch.LongTensor(feat_id)
    return {'id': feat_id}


def decode_libsvm(columns):
    map_func = lambda pair: (int(pair[0]), float(pair[1]))
    # 0 is id, 1 is label
    id, value = zip(*map(lambda col: map_func(col.split(':')), columns[2:]))
    sample = {'id': list(id)}
    return sample



mini_batch = [
    ('4801', '0', '2:1', '4656:1', '5042:1', '5051:1', '5054:1', '5055:1', '5058:1', '5061:1', '5070:1', '5150:1'),
    ('4801', '0', '210:1', '1345:1', '5039:1', '5051:1', '5054:1', '5055:1', '5059:1', '5061:1', '5108:1', '5214:1'),
    ('4801', '0', '276:1', '974:1', '5041:1', '5049:1', '5054:1', '5055:1', '5058:1', '5061:1', '5070:1', '5149:1'),
    ('4801', '0', '117:1', '1998:1', '5041:1', '5046:1', '5053:1', '5055:1', '5058:1', '5061:1', '5100:1', '5192:1'),
    ('4801', '0', '8:1', '2147:1', '5041:1', '5051:1', '5054:1', '5055:1', '5058:1', '5061:1', '5073:1', '5218:1'),
    ('4801', '0', '78:1', '2351:1', '5039:1', '5051:1', '5054:1', '5055:1', '5058:1', '5063:1', '5093:1', '5177:1'),
    ('4801', '0', '773:1', '2645:1', '5045:1', '5047:1', '5053:1', '5055:1', '5058:1', '5067:1', '5069:1', '5258:1'),
    ('4801', '0', '78:1', '4193:1', '5040:1', '5046:1', '5053:1', '5055:1', '5058:1', '5064:1', '5093:1', '5177:1'),
    ('4801', '0', '66:1', '1006:1', '5040:1', '5046:1', '5053:1', '5055:1', '5058:1', '5064:1', '5069:1', '5172:1'),
]
# mini_batch_raw = [
#     [int(item.split(':')[0]) for item in sublist[2:]]
#     for sublist in mini_batch]


mini_batch = mini_batch * 12000
print(len(mini_batch))

begin = time.time()
mini_batch_raw = [
    [int(item.split(':')[0]) for item in sublist[2:]]
    for sublist in mini_batch]
transformed_data = torch.LongTensor(mini_batch_raw)

# pre_processing(mini_batch)
print(time.time() - begin)





