from zoo.orca import init_orca_context
from zoo.ray import RayContext

import ray

sc = init_orca_context(cluster_mode="local", cores=8, memory="20g")
ray_ctx = RayContext(sc=sc, object_store_memory="2g")
ray_ctx.init()


# import ray
# ray.init()

@ray.remote
def f(x):
    return x * x


futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n


counters = [Counter.remote() for i in range(4)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures))
ray_ctx.stop()
