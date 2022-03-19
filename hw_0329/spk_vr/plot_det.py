import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(3, suppress=True)

for result in ["ubm-16-enroll-1.result", "ubm-16-enroll-5.result",
  "ubm-32-enroll-1.result", "ubm-32-enroll-5.result"]:
  with open(result, "r") as f:
    lines = f.readlines()
  lines = [e.strip().split() for e in lines]

  fps = []; fns = []
  scores = [float(e[0]) for e in lines]
  for thres in np.linspace(min(scores), max(scores), 200):
    fp = len([e for e in lines if float(e[0])>thres and e[1]=="nontarget"])
    fn = len([e for e in lines if float(e[0])<thres and e[1]=="target"])
    fps.append(fp/float(len(lines))); fns.append(fn/float(len(lines)))

  plt.plot(fps, fns)
  
plt.xlim([-0.015, 0.2])
plt.ylim([-0.015, 0.2])
plt.legend(["ubm-16-enroll-1", "ubm-16-enroll-5",
  "ubm-32-enroll-1", "ubm-32-enroll-5"])

plt.savefig('det.png')
