"""Bounded useful future RGB batches on CUDA, preserving batch16 and FP32."""
import queue
import threading
import numpy as np
import torch


def ordered_decode(function, keys, initializer, workers=6):
    """Bounded ordered spawn pool; finish outstanding decodes on early close."""
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    import multiprocessing
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initializer,initargs=('egoexo',))
    pending=deque();iterator=iter(keys)
    def fill():
        while len(pending)<workers*2:
            try:key=next(iterator)
            except StopIteration:break
            pending.append(pool.submit(function,key))
    try:
        fill()
        while pending:
            result=pending.popleft().result(timeout=180)
            fill();yield result
    finally:
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)


class RGBPrefetch:
    def __init__(self, decoded, batch_sizes, depth):
        self.decoded=decoded
        self.batch_sizes=list(batch_sizes)
        self.depth=depth
        self.queue=queue.Queue(maxsize=depth)
        self.stop=threading.Event()
        self.error=None
        self.ready=threading.Event()
        self.current=None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # One contiguous reusable ring prevents asynchronous input allocations
        # from pinning fragments inside the encoder's590MiB activation blocks.
        self.storage=torch.empty((depth+2,max(self.batch_sizes),3,16,224,224),device='cuda',dtype=torch.float32)
        self.free=queue.Queue()
        for slot in range(depth+2):self.free.put((slot,None))
        self.thread=threading.Thread(target=self.produce,daemon=True)
        self.thread.start()

    def produce(self):
        torch.cuda.set_device(0)
        stream=torch.cuda.Stream()
        try:
            for count in self.batch_sizes:
                if self.stop.is_set():break
                items=[next(self.decoded) for _ in range(count)]
                rgb=np.stack([v[0] for v in items])
                pts=np.stack([v[1] for v in items])
                while not self.stop.is_set():
                    try:slot,consumed=self.free.get(timeout=.2);break
                    except queue.Empty:pass
                if self.stop.is_set():break
                with torch.inference_mode(),torch.cuda.stream(stream):
                    if consumed is not None:stream.wait_event(consumed)
                    image=self.storage[slot,:count]
                    image.copy_(torch.from_numpy(rgb))
                    event=torch.cuda.Event();event.record(stream)
                item=(slot,pts,event)
                while not self.stop.is_set():
                    try:self.queue.put(item,timeout=.2);break
                    except queue.Full:self.ready.set()
                if self.queue.full():self.ready.set()
            self.ready.set()
        except BaseException as exc:
            self.error=exc;self.ready.set()

    def get(self):
        self.release_current()
        while True:
            if self.error is not None:raise self.error
            try:
                slot,pts,event=self.queue.get(timeout=.2)
                torch.cuda.current_stream().wait_event(event)
                image=self.storage[slot,:len(pts)]
                self.current=slot
                return image,pts
            except queue.Empty:
                if not self.thread.is_alive():raise RuntimeError('RGB prefetch ended before expected batch')

    def release_current(self):
        if self.current is not None:
            event=torch.cuda.Event();event.record(torch.cuda.current_stream())
            self.free.put((self.current,event));self.current=None

    def close(self):
        self.release_current()
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('RGB producer did not stop')
        while not self.queue.empty():
            self.queue.get_nowait()
        torch.cuda.synchronize()
        self.storage=None


def choose_depth(batch_bytes, peak_allocated, capacity_bytes=19*2**30):
    # The producer may hold one batch outside the full queue and one batch can
    # be consumed concurrently. Reserve those plus1GiB for encoder workspace.
    budget=capacity_bytes-peak_allocated-1024*2**20
    return max(1,min(128,int(budget//batch_bytes)-2))
