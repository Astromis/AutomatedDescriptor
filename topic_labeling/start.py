from TopicLableGenerator import *
from TopicModel import LDATopicModel, NMFTopicModel, LSATopicModel
  
lables_gen = TopicLableGenerator('test.txt', ['wordlen', 'tag'], LSATopicModel()) # './chowmein/datasets/nips-2014.dat'

lables_gen.execute(100,8)
lables_gen.print_topical_lables
