from zoo.orca import init_orca_context
from zoo.ray import RayContext

sc = init_orca_context(cluster_mode="local", cores=8, memory="20g")
ray_ctx = RayContext(sc=sc, object_store_memory="5g")
ray_ctx.init()
ray_ctx.stop()
sc.stop()
