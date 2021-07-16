#memoryUtils.py
import torch
import torch.nn as nn

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        self.max_mem = []
        self.cur_mem = []
        self.name = []
        
    def hook_fn(self, module, input, output):
        
        self.max_mem.append(torch.cuda.max_memory_allocated())
        self.cur_mem.append(torch.cuda.memory_allocated())
        self.name.append(get_module_name(str(module)))
        print('Hook is called on {}, max_mem : {}, cur_mem : {}'.format(module, convert_bytes(self.max_mem[-1]), convert_bytes(self.cur_mem[-1])))

def get_module_name(s):
    return s[:s.find('(')]

def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0
    return size