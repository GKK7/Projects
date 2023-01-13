# Import the required module
import psutil


# Get current usage of CPU, RAM and GPU if possible
def get_usage():
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    # GPU discovery may be limited by python version, OS, etc.
    try:
        gpu_percent = psutil.gpu_percent()
    except NotImplementedError:
        gpu_percent = "<Not supported>"
    except AttributeError:
        gpu_percent = "<Inaccessible>"
    print(f"Cpu usage is {cpu_percent} %, RAM usage is {ram_percent} %, GPU usage is {gpu_percent}")


# Find the top CPU consuming program by adding all to a list and sorting based on % usage
def get_top_cpu_user():
    cpu_list = []
    for process in psutil.process_iter():
        try:
            cpu_list.append((process.name(), process.cpu_percent()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    cpu_list = sorted(cpu_list, key=lambda x: x[1], reverse=True)
    print(f"Biggest CPU consumer: {cpu_list[0][0]}, {cpu_list[0][1].__trunc__()} %")


# Find the top RAM consuming program by adding all to a list and sorting based on % usage
def get_top_memory_user():
    ram_list = []
    for process in psutil.process_iter():
        try:
            memory_info = process.memory_info()
            process_info = psutil.Process(process.pid)
            mem_percent = (memory_info.rss / psutil.virtual_memory().total) * 100
            ram_list.append((process.name(), mem_percent))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    ram_list = sorted(ram_list, key=lambda x: x[1], reverse=True)
    print(f"Biggest RAM consumer: {ram_list[0][0]}, {ram_list[0][1].__trunc__()} %")


get_usage()

get_top_cpu_user()

get_top_memory_user()
