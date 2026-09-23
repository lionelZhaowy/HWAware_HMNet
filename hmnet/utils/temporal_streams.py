"""External FP32 memory bank. Detached state is checkpointed independently of weights."""
import torch


class TemporalStreams:
    def __init__(self, backbone, capacity):
        self.memory = backbone.zero_temporal_state(capacity)
        self.last = [None]*capacity
        self.last_rgb = [None]*capacity

    def select(self, metas):
        ids=[int(m["stream_slot"]) for m in metas]
        if len(ids)!=len(set(ids)) or any(i<0 or i>=len(self.last) for i in ids):
            raise ValueError("Invalid or duplicated temporal stream slots")
        reset=[]
        for i,m in zip(ids,metas):
            old=self.last[i]
            reset.append(bool(m["reset"]) or old is None or old[0]!=m["sequence"]
                         or old[2]!=bool(m["flipped"])
                         or not 0 < int(m["curr_time_org"])-old[1] <= 75000)
        indices=torch.tensor(ids,device=self.memory[0].device)
        mask=torch.tensor(reset,device=indices.device).reshape(-1,1,1,1)
        return tuple(s.index_select(0,indices).masked_fill(mask,0.) for s in self.memory)

    def commit(self, metas, current):
        if current is None or len(current)!=len(self.memory):
            raise ValueError("Missing temporal output state")
        ids=torch.tensor([int(m["stream_slot"]) for m in metas],device=self.memory[0].device)
        for bank,value in zip(self.memory,current):
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise FloatingPointError("Temporal state must stay finite FP32")
            bank.index_copy_(0,ids,value.detach())
        for m in metas:
            slot = int(m["stream_slot"])
            self.last[slot]=(m["sequence"],int(m["curr_time_org"]),bool(m["flipped"]))
            if "steps" in m:
                end = m["steps"][-1]
                self.last_rgb[slot] = dict(rgb_id=end["rgb_id"],rgb_time=end["rgb_time"],
                                          rgb_valid=end["rgb_valid"])

    def state_dict(self):
        return dict(memory=[s.detach().cpu() for s in self.memory],last=list(self.last),rgb_sources=list(self.last_rgb))

    def load_state_dict(self, state):
        values=state["memory"]
        if len(values)!=len(self.memory) or len(state["last"])!=len(self.last):
            raise ValueError("Temporal state shape/stream count differs")
        for bank,value in zip(self.memory,values):
            if value.shape!=bank.shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Invalid checkpoint memory")
            bank.copy_(value.to(bank.device))
        self.last=list(state["last"])
        self.last_rgb=list(state.get("rgb_sources",[None]*len(self.last)))
        if len(self.last_rgb) != len(self.last):raise ValueError("RGB source stream count differs")
