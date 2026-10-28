"""Bounded real-batch GPU prefetch; exact sampler order and safe pinned reuse."""
import queue
import threading
import torch


class CudaBatches:
    def __init__(self,data,order,batch,arm,depth):
        self.data,self.order,self.batch,self.arm=data,order,batch,arm
        self.items=queue.Queue(maxsize=depth)
        self.stop=threading.Event()
        self.prefilled=threading.Event()
        self.stream=torch.cuda.Stream()
        self.thread=threading.Thread(target=self.produce,daemon=True)

    def __enter__(self):
        self.thread.start()
        self.prefilled.wait(timeout=30)
        return self

    def put(self,item):
        while not self.stop.is_set():
            try:
                self.items.put(item,timeout=.1)
                if self.items.full():self.prefilled.set()
                return True
            except queue.Full:pass
        return False

    def produce(self):
        try:
            torch.cuda.set_device(0)
            host_events=[None,None]
            with torch.cuda.stream(self.stream):
                for begin in range(0,len(self.order),self.batch):
                    if self.stop.is_set():return
                    slot=self.data.buffer_cursor
                    if host_events[slot] is not None:host_events[slot].synchronize()
                    prepared=self.data.cpu_batch(self.order[begin:begin+self.batch])
                    item=self.data.batch(prepared=prepared,arm=self.arm,transfer_non_blocking=True)
                    event=torch.cuda.Event();event.record(self.stream)
                    host_events[slot]=event
                    if not self.put((item,event)):return
                self.put(None)
        except BaseException as exc:
            self.prefilled.set()
            self.put(exc)
        finally:
            self.prefilled.set()

    def next(self):
        value=self.items.get()
        if isinstance(value,BaseException):raise value
        if value is None:raise StopIteration
        item,event=value
        current=torch.cuda.current_stream();current.wait_event(event)
        for tensor in item.values():
            if torch.is_tensor(tensor) and tensor.is_cuda:tensor.record_stream(current)
        return item

    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('GPU prefetch producer did not terminate')
        self.stream.synchronize()
        while not self.items.empty():
            try:self.items.get_nowait()
            except queue.Empty:break
