""" export script """

import argparse
import numpy as np

import mindspore
from mindspore import context, Tensor, export
from mindspore.train.serialization import load_checkpoint
from src.net import SiameseRPN



def siamrpn_export():
    """ export function """
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        save_graphs=False,
        device_id=args.device_id)
    net = SiameseRPN(groups=1, is_310infer=True)
    load_checkpoint(args.ckpt_file, net=net)
    net.set_train(False)
    input_data1 = Tensor(np.zeros([1, 3, 127, 127]), mindspore.float32)
    input_data2 = Tensor(np.zeros([1, 3, 255, 255]), mindspore.float32)
    input_data = [input_data1, input_data2]
    export(net, *input_data, file_name='siamrpn', file_format="MINDIR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mindspore infering')
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    parser.add_argument('--ckpt_file', type=str, required=True, help='siamRPN ckpt file.')
    args = parser.parse_args()
    siamrpn_export()
