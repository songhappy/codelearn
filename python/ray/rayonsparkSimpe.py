from zoo.orca import init_orca_context
from zoo.ray import RayContext

import ray

sc = init_orca_context(cluster_mode="local", cores=8, memory="20g")
ray_ctx = RayContext(sc=sc, object_store_memory="2g")
ray_ctx.init()


@ray.remote
class TestRay():
    def hostname(self):
        import socket
        return socket.gethostname()

    def ip(self):
        import ray.services as rservices
        return rservices.get_node_ip_address()


actors = [TestRay.remote() for i in range(0, 8)]
print([ray.get(actor.hostname.remote()) for actor in actors])
print([ray.get(actor.ip.remote()) for actor in actors])

ray_ctx.stop()
