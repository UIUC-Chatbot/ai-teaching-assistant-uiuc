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


def get_free_memory_dict(leave_extra_memory_unused_GiB: float = 2, leave_extra_memory_unused_gpu0_GiB: float = 3):
  '''
  Returns a dictionary of GPU IDs and their free memory, in MiB. 
  Compatible with huggingface Accelerate formatting: `max_memory=get_free_memory_dict()`
  
  Accelerate seems to use more memory than we give it, so we default to telling Accelerate we have 2 GiB less than we actually do.
  
  Example output: 
  {0: '24753MiB', 1: '26223MiB', 2: '25603MiB', 3: '9044MiB'}
  '''
  GPUs = GPUtil.getGPUs()
  memory_map = {gpu.id: int(round(gpu.memoryFree)) for gpu in GPUs}
  if leave_extra_memory_unused_GiB > 0:
    for device_id, memory_MiB in memory_map.items():
      memory_map[device_id] = memory_MiB - (leave_extra_memory_unused_GiB * 1024)
  if leave_extra_memory_unused_gpu0_GiB > 0 and 0 in memory_map:
    memory_map[0] = memory_map[0] - (leave_extra_memory_unused_gpu0_GiB * 1024)

  # format to Accelerate's liking
  for device_id, memory_MiB in memory_map.items():
    memory_map[device_id] = f"{int(round(memory_MiB))}MiB"

  return memory_map
