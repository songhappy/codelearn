import logging
import random

logging.basicConfig(level=logging.INFO)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
id_list = [random.randint(1, 30) for _ in range(13)]
n_thread = 4
size = 13 // (n_thread - 1)
print(size)
print(id_list)
for ith in range(n_thread):
    start = ith * size
    end = (ith + 1) * size
    ids = id_list[ith * size: (ith + 1) * size]

    print(ids)
