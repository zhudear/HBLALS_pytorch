from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'dilconv_3x3',
    'resconv_1x1',
    'resconv_3x3',
    'resdilconv_3x3',
    'denseblocks_1x1',
    'denseblocks_3x3'
]
PRIMITIVES2 = [
    'conv_5x5',
    'conv_1x1',
    'conv_3x3',
    'dilconv_3x3',
]
#LOL changeA  skip_connect->conv_3x3  conv_1x1->conv_3x3
LOLcgenotype0 = Genotype(normal=[('denseblocks_3x3', 0), ('denseblocks_1x1', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('denseblocks_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotype1 = Genotype(normal=[('denseblocks_3x3', 0), ('resconv_1x1', 1), ('conv_3x3', 2), ('denseblocks_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('dilconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotype2 = Genotype(normal=[('conv_3x3', 0), ('resconv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('dilconv_3x3', 4), ('denseblocks_1x1', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotype3 = Genotype(normal=[('denseblocks_3x3', 0), ('denseblocks_1x1', 1), ('resconv_1x1', 2), ('denseblocks_3x3', 3), ('conv_3x3', 4), ('denseblocks_1x1', 5), ('denseblocks_1x1', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotype4 = Genotype(normal=[('denseblocks_3x3', 0), ('resconv_1x1', 1), ('conv_3x3', 2), ('denseblocks_1x1', 3), ('denseblocks_3x3', 4), ('resconv_3x3', 5), ('conv_3x3', 6), ('denseblocks_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotype5 = Genotype(normal=[('denseblocks_3x3', 0), ('conv_3x3', 1), ('resconv_3x3', 2), ('denseblocks_3x3', 3), ('denseblocks_1x1', 4), ('denseblocks_1x1', 5), ('dilconv_3x3', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotype6 = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('denseblocks_1x1', 2), ('denseblocks_3x3', 3), ('denseblocks_3x3', 4), ('conv_3x3', 5), ('denseblocks_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotypee = Genotype(normal=[('conv_5x5', 0), ('conv_5x5', 1), ('conv_5x5', 2), ('conv_5x5', 3), ('dilconv_3x3', 4), ('conv_5x5', 5), ('conv_5x5', 6), ('conv_5x5', 7), ('conv_5x5', 8)], normal_concat=None, reduce=None, reduce_concat=None)
LOLcgenotyped = Genotype(normal=[('dilconv_3x3', 0), ('dilconv_3x3', 1), ('dilconv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('dilconv_3x3', 5), ('conv_5x5', 6), ('dilconv_3x3', 7), ('conv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#MIT
MITgenotype0 = Genotype(normal=[('resconv_3x3', 0), ('skip_connect', 1), ('resconv_1x1', 2), ('skip_connect', 3), ('denseblocks_3x3', 4), ('skip_connect', 5), ('skip_connect', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotype1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('resdilconv_3x3', 4), ('denseblocks_3x3', 5), ('conv_1x1', 6), ('denseblocks_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotype2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('resconv_1x1', 4), ('conv_3x3', 5), ('conv_1x1', 6), ('denseblocks_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotype3 = Genotype(normal=[('denseblocks_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('conv_1x1', 5), ('conv_3x3', 6), ('dilconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotype4 = Genotype(normal=[('conv_1x1', 0), ('conv_3x3', 1), ('skip_connect', 2), ('denseblocks_1x1', 3), ('denseblocks_1x1', 4), ('denseblocks_1x1', 5), ('conv_1x1', 6), ('denseblocks_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotype5 = Genotype(normal=[('conv_1x1', 0), ('conv_3x3', 1), ('skip_connect', 2), ('conv_1x1', 3), ('denseblocks_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotype6 = Genotype(normal=[('dilconv_3x3', 0), ('resdilconv_3x3', 1), ('skip_connect', 2), ('denseblocks_3x3', 3), ('conv_1x1', 4), ('denseblocks_1x1', 5), ('conv_1x1', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotypee = Genotype(normal=[('conv_1x1', 0), ('conv_5x5', 1), ('conv_5x5', 2), ('conv_1x1', 3), ('conv_3x3', 4), ('conv_5x5', 5), ('conv_5x5', 6), ('conv_1x1', 7), ('conv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
MITgenotyped = Genotype(normal=[('conv_1x1', 0), ('dilconv_3x3', 1), ('conv_5x5', 2), ('conv_1x1', 3), ('conv_3x3', 4), ('dilconv_3x3', 5), ('conv_5x5', 6), ('conv_1x1', 7), ('dilconv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#LOL 1-c
LOLgenotype01c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype11c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype21c= Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype31c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype41c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype51c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype61c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotypee1c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7),('conv_1x1', 8)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotyped1c = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7), ('conv_1x1', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#LOL 1-rc
LOLgenotype01rc = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resconv_1x1', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('resconv_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype11rc = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resconv_1x1', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('resconv_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype21rc= Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resconv_1x1', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('resconv_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype31rc = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resconv_1x1', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('resconv_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype41rc = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resconv_1x1', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('resconv_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype51rc = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resconv_1x1', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('resconv_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype61rc = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resconv_1x1', 2), ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5), ('resconv_1x1', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotypee1rc = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7),('conv_1x1', 8)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotyped1rc = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('conv_1x1', 4), ('conv_1x1', 5), ('conv_1x1', 6), ('conv_1x1', 7), ('conv_1x1', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#LOL 3c
NRM0noc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
NRM1noc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
NRM2noc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
NRM3noc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
NRM4noc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
NRM5noc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
NRM6noc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
encodernoc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7), ('conv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
decodernoc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7), ('conv_3x3',  8)], normal_concat=None, reduce=None, reduce_concat=None)
#LOL 3-rc
LOLgenotype03rc = Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype13rc = Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype23rc= Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype33rc = Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype43rc = Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype53rc = Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype63rc = Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotypee3rc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7), ('conv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotyped3rc = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_3x3', 5), ('conv_3x3', 6), ('conv_3x3', 7), ('conv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#LOL loss A
LAgenotype0 = Genotype(normal=[('denseblocks_3x3', 0), ('denseblocks_3x3', 1), ('skip_connect', 2), ('denseblocks_3x3', 3), ('dilconv_3x3', 4), ('skip_connect', 5), ('resconv_3x3', 6), ('denseblocks_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotype1 = Genotype(normal=[('denseblocks_3x3', 0), ('resconv_3x3', 1), ('resconv_3x3', 2), ('conv_1x1', 3), ('denseblocks_1x1', 4), ('denseblocks_3x3', 5), ('denseblocks_3x3', 6), ('dilconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotype2 = Genotype(normal=[('conv_1x1', 0), ('denseblocks_1x1', 1), ('conv_1x1', 2), ('dilconv_3x3', 3), ('denseblocks_3x3', 4), ('denseblocks_1x1', 5), ('conv_3x3', 6), ('dilconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotype3 = Genotype(normal=[('resconv_1x1', 0), ('denseblocks_1x1', 1), ('denseblocks_1x1', 2), ('denseblocks_3x3', 3), ('conv_1x1', 4), ('denseblocks_1x1', 5), ('conv_1x1', 6), ('denseblocks_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotype4 = Genotype(normal=[('denseblocks_1x1', 0), ('denseblocks_3x3', 1), ('conv_3x3', 2), ('conv_1x1', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('conv_1x1', 6), ('denseblocks_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotype5 = Genotype(normal=[('denseblocks_3x3', 0), ('conv_3x3', 1), ('denseblocks_3x3', 2), ('conv_3x3', 3), ('denseblocks_3x3', 4), ('resconv_1x1', 5), ('denseblocks_1x1', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotype6 = Genotype(normal=[('resdilconv_3x3', 0), ('conv_3x3', 1), ('dilconv_3x3', 2), ('denseblocks_3x3', 3), ('denseblocks_3x3', 4), ('denseblocks_1x1', 5), ('resconv_1x1', 6), ('conv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotypee = Genotype(normal=[('conv_5x5', 0), ('conv_5x5', 1), ('dilconv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('conv_5x5', 5), ('conv_5x5', 6), ('conv_5x5', 7), ('conv_5x5', 8)], normal_concat=None, reduce=None, reduce_concat=None)
LAgenotyped = Genotype(normal=[('dilconv_3x3', 0), ('conv_1x1', 1), ('dilconv_3x3', 2), ('conv_1x1', 3), ('conv_5x5', 4), ('conv_1x1', 5), ('dilconv_3x3', 6), ('conv_1x1', 7), ('conv_1x1', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#LOL
LOLgenotype0 = Genotype(normal=[('conv_3x3', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_1x1', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype1 = Genotype(normal=[('resdilconv_3x3', 0), ('conv_3x3', 1), ('denseblocks_3x3', 2), ('dilconv_3x3', 3), ('skip_connect', 4), ('skip_connect', 5), ('resconv_3x3', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype2 = Genotype(normal=[('conv_1x1', 0), ('dilconv_3x3', 1), ('denseblocks_1x1', 2), ('denseblocks_3x3', 3), ('skip_connect', 4), ('resdilconv_3x3', 5), ('resconv_3x3', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype3 = Genotype(normal=[('denseblocks_1x1', 0), ('denseblocks_3x3', 1), ('denseblocks_3x3', 2), ('conv_1x1', 3), ('resconv_1x1', 4), ('resdilconv_3x3', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype4 = Genotype(normal=[('dilconv_3x3', 0), ('conv_1x1', 1), ('denseblocks_1x1', 2), ('conv_1x1', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype5 = Genotype(normal=[('denseblocks_3x3', 0), ('dilconv_3x3', 1), ('dilconv_3x3', 2), ('conv_1x1', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotype6 = Genotype(normal=[('denseblocks_3x3', 0), ('conv_1x1', 1), ('denseblocks_3x3', 2), ('conv_3x3', 3), ('resdilconv_3x3', 4), ('skip_connect', 5), ('resconv_3x3', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotypee = Genotype(normal=[('dilconv_3x3', 0), ('dilconv_3x3', 1), ('conv_5x5', 2), ('dilconv_3x3', 3), ('dilconv_3x3', 4), ('conv_5x5', 5), ('conv_3x3', 6), ('conv_3x3', 7), ('dilconv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
LOLgenotyped = Genotype(normal=[('conv_3x3', 0), ('conv_5x5', 1), ('conv_3x3', 2), ('conv_3x3', 3), ('conv_5x5', 4), ('conv_5x5', 5), ('conv_5x5', 6), ('conv_5x5', 7), ('conv_1x1', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#SIHR
SIHRgenotype0 = Genotype(normal=[('resconv_3x3', 0), ('resconv_3x3', 1), ('skip_connect', 2), ('conv_3x3', 3), ('resconv_3x3', 4), ('skip_connect', 5), ('resconv_3x3', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotype1 = Genotype(normal=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_1x1', 2), ('denseblocks_1x1', 3), ('resconv_1x1', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotype2 = Genotype(normal=[('conv_3x3', 0), ('conv_1x1', 1), ('dilconv_3x3', 2), ('resconv_1x1', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resdilconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotype3 = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('dilconv_3x3', 2), ('denseblocks_1x1', 3), ('resconv_3x3', 4), ('resconv_3x3', 5), ('resdilconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotype4 = Genotype(normal=[('dilconv_3x3', 0), ('conv_1x1', 1), ('conv_3x3', 2), ('conv_1x1', 3), ('resdilconv_3x3', 4), ('resconv_3x3', 5), ('resdilconv_3x3', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotype5 = Genotype(normal=[('conv_3x3', 0), ('denseblocks_3x3', 1), ('resconv_1x1', 2), ('denseblocks_1x1', 3), ('resdilconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotype6 = Genotype(normal=[('resdilconv_3x3', 0), ('denseblocks_3x3', 1), ('resconv_3x3', 2), ('conv_1x1', 3), ('resdilconv_3x3', 4), ('resconv_3x3', 5), ('resconv_3x3', 6), ('resconv_3x3', 7)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotypee = Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_3x3', 3), ('conv_5x5', 4), ('conv_3x3', 5), ('conv_1x1', 6), ('conv_1x1', 7), ('dilconv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
SIHRgenotyped = Genotype(normal=[('conv_3x3', 0), ('conv_5x5', 1), ('dilconv_3x3', 2), ('dilconv_3x3', 3), ('conv_5x5', 4), ('conv_3x3', 5), ('dilconv_3x3', 6), ('conv_5x5', 7), ('dilconv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
#UIE
UIEgenotype1 = Genotype(normal=[('resconv_1x1', 0), ('resconv_1x1', 1), ('resdilconv_3x3', 2), ('denseblocks_1x1', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotype2 = Genotype(normal=[('denseblocks_3x3', 0), ('conv_1x1', 1), ('dilconv_3x3', 2), ('resconv_3x3', 3), ('skip_connect', 4), ('skip_connect', 5), ('resdilconv_3x3', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotype3 = Genotype(normal=[('denseblocks_3x3', 0), ('conv_3x3', 1), ('conv_1x1', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotype4 = Genotype(normal=[('conv_1x1', 0), ('conv_3x3', 1), ('dilconv_3x3', 2), ('dilconv_3x3', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotype5 = Genotype(normal=[('denseblocks_3x3', 0), ('dilconv_3x3', 1), ('denseblocks_1x1', 2), ('dilconv_3x3', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6), ('resconv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotype6 = Genotype(normal=[('denseblocks_1x1', 0), ('conv_1x1', 1), ('dilconv_3x3', 2), ('dilconv_3x3', 3), ('skip_connect', 4), ('resdilconv_3x3', 5), ('skip_connect', 6), ('skip_connect', 7)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotypee = Genotype(normal=[('dilconv_3x3', 0), ('conv_1x1', 1), ('conv_5x5', 2), ('conv_5x5', 3), ('dilconv_3x3', 4), ('conv_1x1', 5), ('conv_5x5', 6), ('conv_3x3', 7), ('dilconv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotyped = Genotype(normal=[('conv_3x3', 0), ('conv_5x5', 1), ('conv_5x5', 2), ('conv_5x5', 3), ('conv_1x1', 4), ('conv_5x5', 5), ('dilconv_3x3', 6), ('conv_3x3', 7), ('dilconv_3x3', 8)], normal_concat=None, reduce=None, reduce_concat=None)
UIEgenotype0 = Genotype(normal=[('conv_1x1', 0), ('skip_connect', 1), ('resconv_3x3', 2), ('skip_connect', 3), ('denseblocks_1x1', 4), ('denseblocks_3x3', 5), ('skip_connect', 6), ('conv_1x1', 7)], normal_concat=None, reduce=None, reduce_concat=None)

