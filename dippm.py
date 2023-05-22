import argparse
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Batch
import numpy as np
import tvm
from tvm import relay
import networkx as nx


ops_code = {'nn.conv2d': 1, 'nn.conv2d_transpose': 2, 'nn.dense': 3, 'nn.batch_matmul':4, 'nn.bias_add': 5, 'nn.softmax':6, 'nn.dropout': 7, 
            'nn.layer_norm':8, 'reshape':9, 'transpose':10 , 'add': 11, 'split': 12, 'multiply': 13, 'squeeze': 14, 'nn.adaptive_avg_pool2d':15,
            'nn.max_pool2d': 16}
FEATURE_VECTOR_LEN = 12
SHAPE = 4
OPS_ONE_HOT_ENCODE_LEN = len(ops_code)

def create_graph(expr):
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)
    node_dict = {}
    tvm.relay.analysis.post_order_visit(expr, lambda x: _traverse_expr(x, node_dict))
    G = nx.DiGraph()
    nodes_list = []
    for node, node_id in sorted(node_dict.items(), key=lambda x: x[1]):
        if isinstance(node, tvm.relay.Call):
            if isinstance(node.op, tvm.ir.Op):
                shape_list = []
                for arg in node.args:
                    if isinstance(arg, tvm.relay.Var):
                        try:
                            shape = np.array(list([int(x) for x in arg.type_annotation.shape]), dtype='float32')
                            shape_list.append(shape)
                        except:
                            continue
                if shape_list:
                    shape = shape_list[-1]
                    if shape.size > SHAPE: shape = shape[:SHAPE]
                    if shape.size < SHAPE: shape = np.concatenate((shape, np.zeros(SHAPE - shape.size, dtype="float32")))
                    if node.op.name not in ops_code:
                        op = np.zeros(OPS_ONE_HOT_ENCODE_LEN, dtype="float32")
                    else:
                        op = np.eye(OPS_ONE_HOT_ENCODE_LEN, dtype="float32")[ops_code[node.op.name]]
                    attrs = {k:getattr(node.attrs, k) for k in node.attrs.keys()} if hasattr(node.attrs, 'keys') else {}
                    # print(node_id, node.op.name, shape, attrs)
                    attr_vector = np.zeros(12, dtype="float32")
                    if 'conv2d' in node.op.name:
                        attr_vector = np.array(list([int(x) for x in attrs['strides']]) + list([int(x) for x in attrs['padding']]) + list([int(x) for x in attrs['dilation']]) + [int(attrs['groups'])] + [int(attrs['channels'])] + list([int(x) for x in attrs['kernel_size']]), dtype="float32")
                    if attr_vector.size > FEATURE_VECTOR_LEN:
                        attr_vector = attr_vector[:FEATURE_VECTOR_LEN]
                    if attr_vector.size < FEATURE_VECTOR_LEN:
                        attr_vector = np.concatenate((attr_vector, np.zeros(FEATURE_VECTOR_LEN - attr_vector.size, dtype="float32")))
                    features_vector = np.concatenate((op, attr_vector, shape), dtype="float32")
                    G.add_node(str(node_id), attributes=features_vector)
                    nodes_list.append(str(node_id))

    for i in range(len(nodes_list) - 1):
        G.add_edge(nodes_list[i], nodes_list[i+1])
    return G

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def pytorch(model, batch, size):
    inputs = torch.randn((batch, *size))
    input_name = "input0"
    scripted_model = torch.jit.trace(model, inputs).eval()
    shape_list = [(input_name, inputs.shape)]
    mod, _ = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod

def ppm(path, data):
    from model import Model
    model = Model()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    batch = Batch()
    batch = batch.from_data_list([data])
    out = model(batch)
    ps = out.data.cpu().numpy()[0]
    time_ms = (ps[0] / 1000).round(3)
    power_j = (ps[1] / 10).round(3)
    mem_mb = int(ps[2])
    mig = "Undefined"
    if 20000 < mem_mb < 40000: mig = "7g.40gb"
    elif 10000 < mem_mb < 20000: mig = "3g.20gb"
    elif 5000 < mem_mb < 10000: mig = "2g.10gb"
    elif 0 < mem_mb < 5000: mig = "1g.5gb"
    return mem_mb, power_j, time_ms, mig

def predict(model, batch, input, device):
    input = input.split(",")
    input_size = (int(input[0]), int(input[1]), int(input[2]))
    
    mod = pytorch(model, batch, input_size)
        
    G = create_graph(mod["main"])
    func = run_opt_pass(mod['main'], relay.transform.InferType())
    mac = int(relay.analysis.get_total_mac_number(func))/1e9
    dtype = relay.analysis.all_dtypes(func)
    ops_freq = relay.analysis.list_op_freqs(mod)
    G.graph["static"] = np.array([batch, mac, dtype, input_size, ops_freq], dtype="object") # type: ignore
    if "nn.relu" in G.graph['static'][4]: relu = int(G.graph['static'][4]["nn.relu"]) # type: ignore
    else: relu = 0
    G.graph['static'] = np.array([G.graph['static'][0], G.graph['static'][1], int(G.graph['static'][4]["nn.conv2d"]), int(G.graph['static'][4]["nn.dense"]), relu], dtype="float32") # type: ignore
    data = from_networkx(G, all)
    mem_mb, power_j, time_ms, mig = ppm("models/epoch=487-step=3589240.ckpt", data)

    return (mem_mb, power_j, time_ms, mig)
    















# efficientnet_b0, Batch 32, NVIDIA A100-SXM4-40GB, juwelsdevelbooster,
# Actual Memory Consumption 4772 mb 
# Actual Inference Time 80.49 ms 

