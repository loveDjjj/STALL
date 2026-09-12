"""按依赖启动开发四格；不自动扩张专家超参数。"""
import subprocess,sys
from pathlib import Path
from mainline_experts.run import configuration


def execute(root,args=None):
    root=Path(root)
    def run(*a):subprocess.run([sys.executable,'-m','src.mainline_experts.run',*a],cwd=root,check=True)
    for d in configuration(root)['datasets']:run('fit','--dataset',d)
    run('prepare');run('verify','--pilot');run('dense','--pilot','--rank','0','--world-size','1')
    jobs=[subprocess.Popen([sys.executable,'-m','src.mainline_experts.run','dense','--rank',str(i),'--world-size','2'],cwd=root) for i in (0,1)]
    codes=[p.wait() for p in jobs]
    if any(codes):raise RuntimeError('密集评分失败，保留检查点后排查')
    run('evaluate');run('analyze');run('verify')
    subprocess.run([sys.executable,'-m','pytest','tests','-q','--junitxml='+str(root/configuration(root)['run_directory']/'tests.xml')],cwd=root,check=True)
    run('report')
