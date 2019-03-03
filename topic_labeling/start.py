from TopicLableGenerator import *

  
lables_gen = TopicLableGenerator('./chowmein/datasets/nips-2014.dat', ['wordlen', 'tag'])

lables_gen.execute(100,8)
lables_gen.print_topical_lables