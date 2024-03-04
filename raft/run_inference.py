import os
import glob as gb

data_path = "../data/DAVIS2016"
# gap = [1, 2]
gap=[1]
reverse = [0, 1]
rgbpath = data_path + '/JPEGImages/480p'  # path to the dataset
folder = gb.glob(os.path.join(rgbpath, '*').replace("\\","/"))
for r in reverse:
  for g in gap:
    for f in folder:
      f='horsejump-high'
      print('===> Runing {}, gap {}'.format(f, g))
      model = '../models/raft-things.pth'  # model
      outroot = ""
      raw_outroot = ""
      if r == 1:
        raw_outroot = data_path + '/Flows_gap-{}/'.format(g)  # where to raw flow
        outroot = data_path + '/FlowImages_gap-{}/'.format(g)  # where to save the image flow
        # frames = gb.glob(os.path.join(f, '*'))
      elif r == 0:
        raw_outroot = data_path + '/Flows_gap{}/'.format(g)  # where to raw flow
        outroot = data_path + '/FlowImages_gap{}/'.format(g)  # where to save the image flow
        # frames = gb.glob(os.path.join(f, '*'))
      os.system("python predict.py "
                "--gap {} --model {} --path {} "
                "--outroot {} --reverse {} --raw_outroot {}".format(g, model, f, outroot, r, raw_outroot))
      # for fr in frames:
      #   os.system("python predict.py "
      #             "--gap {} --model {} --path {} "
      #             "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, fr, outroot, r, raw_outroot))
