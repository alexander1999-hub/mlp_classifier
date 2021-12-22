import pickle 
import torch
from mlp import process_output
from mlp import two_layers

net = two_layers(input_size=108, hidden_size=36, output_size=4).to(device='cuda:0')
checkpoint = torch.load('mlp_4cls_3_layers_slice_3_hm.pt')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
sep_head_tensor = torch.squeeze(torch.load('sep_head0.pt')['hm']).permute(1,2,0)
with open('prediction.pkl', 'rb') as f:
        output = pickle.load(f)
output_dict = output['seq_0_frame_0.pkl']

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()

result = process_output(output_dict, sep_head_tensor, 3, net)

ender.record()
torch.cuda.synchronize()
print(starter.elapsed_time(ender), " ms for ", len(output_dict['box3d_lidar']), " objects")

print(len(result['box3d_lidar']), " objects left")
