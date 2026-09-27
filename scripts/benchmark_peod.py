#!/usr/bin/env python3
"""Bounded real-data PEOD trainer benchmarks, with 1-second resource sampling."""
import argparse,json,os,subprocess,time,threading
from pathlib import Path
import numpy as np
import psutil
from hmnet.models.efficientvit_tasks import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batches',default='32,64,96,128');p.add_argument('--workers',default='2');p.add_argument('--steps',type=int,default=4)
    p.add_argument('--modality',default='rgbdvs');p.add_argument('--representation',default='rvt_histogram')
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);reports=[]
    for batch in map(int,a.batches.split(',')):
        for workers in map(int,a.workers.split(',')):
            dest=out/f'b{batch}_w{workers}';dest.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
            command=[str(ROOT/'scripts/hmnet-python'),str(ROOT/'scripts/task_temporal.py'),'train','--modality',a.modality,
                '--representation',a.representation,'--data-root',a.data_root,'--batch',str(batch),'--workers',str(workers),
                '--stop-after',str(a.steps),'--output',str(dest)]
            with (dest/'console.log').open('w') as log,(dest/'resources.jsonl').open('w') as resource:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    now=time.time();mem=psutil.virtual_memory();io=psutil.disk_io_counters(perdisk=True).get('sdb')
                    raw=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
                    record=dict(time=now,gpu_util=float(raw[0]),gpu_memory_mib=float(raw[1]),cpu_percent=psutil.cpu_percent(),
                        ram_available=mem.available,io=io._asdict() if io else {})
                    resource.write(json.dumps(record)+'\n');resource.flush();time.sleep(1)
            report=dict(batch=batch,workers=workers,modality=a.modality,representation=a.representation,exit_code=process.returncode)
            if process.returncode==0:
                values=[json.loads(l) for l in (dest/'metrics.jsonl').read_text().splitlines()][a.warmup:]
                resources=[json.loads(l) for l in (dest/'resources.jsonl').read_text().splitlines()]
                timed=[r for r in resources if values[0].get('wall_time',0)-values[0]['seconds']<=r['time']<=values[-1].get('wall_time',float('inf'))]
                if timed:resources=timed
                # GPU event elapsed includes stream idle gaps during CPU-side loss/launch work.
                seconds=sum(v['seconds'] for v in values)
                report.update(updates=len(values),seconds=seconds,samples_per_second=sum(v['samples'] for v in values)/seconds,
                    step_median=float(np.median([v['seconds'] for v in values])),step_p95=float(np.quantile([v['seconds'] for v in values],.95)),
                    data_wait_mean=float(np.mean([v['data_wait_seconds'] for v in values])),
                    peak_allocated_mib=max(v['peak_memory_mib'] for v in values),peak_reserved_mib=max(v['peak_reserved_mib'] for v in values),
                    gpu_elapsed_ms_mean=float(np.mean([v.get('gpu_elapsed_ms',float('nan')) for v in values])),
                    gpu_util_mean=float(np.mean([r['gpu_util'] for r in resources])),gpu_util_range=[min(r['gpu_util'] for r in resources),max(r['gpu_util'] for r in resources)],
                    steady_criterion=len(values)>=100 and seconds>=120,cache_scope='see data manifest; pilot is NOT full-training I/O evidence')
                # Checkpoint save is exercised, then remove only this disposable benchmark's large file.
                (dest/'checkpoint.pth').unlink()
            reports.append(report);(out/'report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(report),flush=True)
            if process.returncode:break

if __name__=='__main__':main()
