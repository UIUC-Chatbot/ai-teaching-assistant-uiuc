import GPUtil  # pip install gputil


def get_gpu_ids_with_sufficient_memory(memory_requirement_GB):
    '''
    Returns the MINIMAL SET of GPU IDs that, combined, have at least `memory_requirement` MB of free memory.
    You will need to use all returned GPU IDs to get the desired memory requirement.
    It returns lower IDs first [0, 1, ...]
    
    If `memory_requirement` is 0, returns all available GPUs.
    If `memory_requirement` is not available, returns an empty list.
    '''
    memory_requirement_MB = float(memory_requirement_GB * 1024)
    GPUs = sorted(GPUtil.getGPUs(), key=lambda x: x.memoryFree, reverse=True)
    total_memory = sum(gpu.memoryFree for gpu in GPUs)
    if memory_requirement_MB > total_memory:
        return []
    GPU_IDs = []
    for gpu in GPUs:
        if memory_requirement_MB <= 0:
            break
        GPU_IDs.append(gpu.id)
        memory_requirement_MB -= gpu.memoryFree
    return GPU_IDs


def get_device_with_most_free_memory():
    '''
    Returns the GPU ID of the GPU with the most free memory.
    '''
    GPUs = GPUtil.getGPUs()
    return sorted(GPUs, key=lambda x: x.memoryFree, reverse=True)[0].id


def get_free_memory_dict():
    '''
    Returns a dictionary of GPU IDs and their free memory.
    '''
    GPUs = GPUtil.getGPUs()
    return {gpu.id: gpu.memoryFree for gpu in GPUs}