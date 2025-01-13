# dunder methods
print(dir(int))

# lambda functions 


# yield


# Understanding __get__ and __set__ and Python descriptors
def func(a, b):
    return a + b

print(func.__get__(1, 2))


def convert_forward(m, target_m, new_forward):
    for _, sub_m in m.named_children():
        if sub_m.__class__ == target_m:  #target_m is a module
            bound_method = new_forward.__get__(sub_m, sub_m.__class__)  #new_forward
            setattr(sub_m, "forward", bound_method)
        convert_forward(sub_m, target_m, new_forward)

# for sub modules in model, if it is a Resampler, replace its forward method with qwen_vl_resampler_forward
# convert_forward(model,
#                 visual_module.Resampler, 
#                 qwen_vl_resampler_forward
#                 )