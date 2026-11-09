import sys, subprocess, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0); print("gpu", p.name, round(p.total_memory/2**30,1), "GB")
print(subprocess.run(["bash","-c","free -g | head -2; echo cpus $(nproc); df -h /content | tail -1; pip list 2>/dev/null | grep -iE '^(timm|torchvision|scikit-learn|pandas|pyyaml) '"],capture_output=True,text=True).stdout)
