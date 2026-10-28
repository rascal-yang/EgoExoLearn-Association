"""Contiguous FP16 prefix ring; bounded real future clips and CUDA events."""
import queue
import threading
import numpy as np
import torch


class PrefixRing:
    def __init__(self,data,ids,batch,depth):
        self.data,self.ids,self.batch,self.depth=data,list(ids),batch,depth
        self.queue=queue.Queue(maxsize=depth);self.free=queue.Queue()
        self.stop=threading.Event();self.error=None;self.current=None
        torch.cuda.synchronize();torch.cuda.empty_cache()
        self.storage=torch.empty((depth+2,batch,3137,768),dtype=torch.float16,device='cuda')
        for slot in range(depth+2):self.free.put((slot,None))
        self.thread=threading.Thread(target=self.produce,daemon=True);self.thread.start()

    def produce(self):
        torch.cuda.set_device(0);stream=torch.cuda.Stream()
        try:
            for begin in range(0,len(self.ids),self.batch):
                if self.stop.is_set():break
                ids,x,mask=self.data.cpu_batch(self.ids[begin:begin+self.batch])
                while not self.stop.is_set():
                    try:slot,event=self.free.get(timeout=.2);break
                    except queue.Empty:pass
                if self.stop.is_set():break
                with torch.inference_mode(),torch.cuda.stream(stream):
                    if event is not None:stream.wait_event(event)
                    self.storage[slot,:len(ids)].copy_(torch.from_numpy(x))
                    ready=torch.cuda.Event();ready.record(stream)
                item=(slot,ids,mask,ready)
                while not self.stop.is_set():
                    try:self.queue.put(item,timeout=.2);break
                    except queue.Full:pass
        except BaseException as exc:self.error=exc

    def release(self):
        if self.current is not None:
            event=torch.cuda.Event();event.record(torch.cuda.current_stream())
            self.free.put((self.current,event));self.current=None

    def get(self):
        self.release()
        while True:
            if self.error is not None:raise self.error
            try:
                slot,ids,mask,event=self.queue.get(timeout=.2)
                torch.cuda.current_stream().wait_event(event);self.current=slot
                return dict(ids=ids,x=self.storage[slot,:len(ids)],mask=torch.as_tensor(mask,device='cuda'))
            except queue.Empty:
                if not self.thread.is_alive():raise RuntimeError('Prefix ring ended before expected batch')

    def close(self):
        self.release();self.stop.set();self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('Prefix producer did not stop')
        torch.cuda.synchronize();self.storage=None
