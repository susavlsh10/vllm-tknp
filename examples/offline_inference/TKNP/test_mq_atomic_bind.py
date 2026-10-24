"""Exercise the actual MessageQueue remote-bind block without loading CUDA."""
import ast
from concurrent.futures import ThreadPoolExecutor
import ipaddress
from pathlib import Path
import threading
from types import SimpleNamespace
import zmq


def main():
    source=Path(__file__).resolve().parents[3]/'vllm/distributed/device_communicators/shm_broadcast.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MessageQueue')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    block=next(n for n in init.body if isinstance(n,ast.If) and ast.unparse(n.test)=='n_remote_reader > 0')
    code=compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec')
    for iteration in range(4):
        barrier=threading.Barrier(8)
        def worker(_):
            ctx=zmq.Context();sockets=[];ports=[]
            try:
                for i in range(8):
                    obj=SimpleNamespace()
                    ns=dict(self=obj,context=ctx,n_remote_reader=1,connect_ip='127.0.0.1',
                        remote_addr_ipv6=False,XPUB=zmq.XPUB,XPUB_VERBOSE=zmq.XPUB_VERBOSE,
                        IPV6=zmq.IPV6,is_valid_ipv6_address=lambda ip:ipaddress.ip_address(ip).version==6)
                    exec(code,ns)
                    sock=obj.remote_socket;sockets.append(sock)
                    address=ns['remote_subscribe_addr']
                    assert sock.getsockopt_string(zmq.LAST_ENDPOINT)==address
                    ports.append(int(address.rsplit(':',1)[1]))
                barrier.wait(timeout=20)
                return ports
            finally:
                for sock in sockets:sock.close(linger=0)
                ctx.destroy(linger=0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            ports=sum(pool.map(worker,range(8)),[])
        assert len(ports)==len(set(ports))==64
    print('ATOMIC_BIND_PASS 4 rounds x64 simultaneously held queue sockets; advertised endpoints verified')


if __name__=='__main__':main()
