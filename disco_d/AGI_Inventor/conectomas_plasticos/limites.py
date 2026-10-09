"""Limite real do job Windows: agregado 6 GiB, até três processos filhos incluídos."""
import ctypes
import sys
import time

HANDLES = []


def constrain(memory_gib=6, processes=3):
    if sys.platform != 'win32':
        raise RuntimeError('Este lançador de limite foi implementado para Windows')
    from ctypes import wintypes as w
    size = ctypes.c_size_t
    class Basic(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                    ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', size), ('MaximumWorkingSetSize', size),
                    ('ActiveProcessLimit', w.DWORD), ('Affinity', size), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(x, ctypes.c_ulonglong) for x in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                     'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
    class Extended(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO), ('ProcessMemoryLimit', size),
                    ('JobMemoryLimit', size), ('PeakProcessMemoryUsed', size), ('PeakJobMemoryUsed', size)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes, kernel.CreateJobObjectW.restype = [ctypes.c_void_p, w.LPCWSTR], w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.SetInformationJobObject.restype = w.BOOL
    kernel.AssignProcessToJobObject.argtypes, kernel.AssignProcessToJobObject.restype = [w.HANDLE, w.HANDLE], w.BOOL
    kernel.GetCurrentProcess.restype = w.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x200 | 0x8
    info.BasicLimitInformation.ActiveProcessLimit = processes
    info.JobMemoryLimit = memory_gib * 1024**3
    if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    HANDLES.append(job)
    return {'job_memory_bytes': info.JobMemoryLimit, 'active_process_limit': processes, 'enforced': True}


class Budget:
    def __init__(self, seconds):
        if not 1 <= seconds <= 1800:
            raise ValueError('Piloto limitado a 30 minutos')
        self.start = time.monotonic()
        self.deadline = self.start + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Orçamento do piloto esgotado')
