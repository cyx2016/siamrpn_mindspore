"""mxBase preprocess , In order to adapt to mxbase, compatibility is not good """
import os
import numpy as np

def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    """ generate anchors """
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def generateConfigBin():
    """ get anchors and hanning for eval """
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size 255
    total_stride = 8                       # total stride of backbone
    anchor_base_size = 8
    valid_scope = int((instance_size - exemplar_size) / total_stride / 2)
    valid_scope = 2 * valid_scope + 1
    anchor_scales = np.array([8,])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    score_size = int((instance_size - exemplar_size) / 8 + 1)
    anchors = generate_anchors(total_stride, anchor_base_size, anchor_scales,
                               anchor_ratios, valid_scope)

    windows = np.tile(np.outer(np.hanning(score_size), np.hanning(score_size))[None, :],
                      [anchor_num, 1, 1]).flatten()
    path1 = os.path.join(os.getcwd(), "src", 'anchors.bin')
    path2 = os.path.join(os.getcwd(), "src", 'windows.bin')
    if os.path.exists(path1):
        os.remove(path1)
    if os.path.exists(path2):
        os.remove(path2)
    anchors.tofile(path1)
    windows.astype(np.float32).tofile(path2)

if __name__ == '__main__':
    generateConfigBin()
