import controlnet.config as cf
from controlnet.cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if cf.save_memory:
    enable_sliced_attention()
